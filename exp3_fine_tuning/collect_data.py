"""
Collect teacher responses for offline SFT / distillation.

Uses an OpenAI-compatible chat API (remote LLM or a local vLLM OpenAI server)
to sample solutions on math questions, then writes SFT-ready JSON:
  [{"question": "...", "golden_res": "..."}, ...]

Example (local vLLM OpenAI server):
  python /root/hidden_prob/exp3_fine_tuning/collect_data.py \
      --base_url http://127.0.0.1:8000/v1 \
      --api_key EMPTY \
      --model Qwen2.5-32B-Instruct \
      --data_path /root/hidden_prob/data/math/train_split3000.json \
      --save_path /root/autodl-tmp/exp3_sft/teacher/math_en_sft.json \
      --language_type en \
      --n 1 \
      --temperature 0.7 \
      --max_tokens 2048

Example (pick 1 of n samples as golden_res; keep raw dump):
  python /root/hidden_prob/exp3_fine_tuning/collect_data.py \
      --base_url http://127.0.0.1:8000/v1 \
      --api_key EMPTY \
      --model Qwen2.5-32B-Instruct \
      --data_path /root/hidden_prob/data/math/train_split3000.json \
      --save_path /root/autodl-tmp/exp3_sft/teacher/math_zh_sft.json \
      --raw_save_path /root/autodl-tmp/exp3_sft/teacher/math_zh_raw.json \
      --language_type zh \
      --n 4 \
      --pick first
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exp1_math.language import Language


@dataclass
class Config:
    data_path: str
    save_path: str
    raw_save_path: str | None
    language_type: str
    model: str
    n: int
    temperature: float
    top_p: float
    max_tokens: int
    max_workers: int
    max_retries: int
    pick: str
    limit: int | None
    resume: bool


class Client:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        config: Config,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.config = config
        self.language = Language(config.language_type)
        self._build_client()
        self.load_data()

    def _build_client(self):
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def load_data(self):
        with open(self.config.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list) and len(data) > 0, f"empty data: {self.config.data_path}"

        self.items: list[dict] = []
        for i, row in enumerate(data):
            item = {
                "question": row["question"],
                "answer": row['answer'],
            }
            # keep useful metadata if present
            for k in ("subject", "level"):
                if k in row:
                    item[k] = row[k]
            self.items.append(item)

        if self.config.limit is not None:
            self.items = self.items[: self.config.limit]

        self.messages_ls = self.format_prompt(
            [x["question"] for x in self.items],
            self.config.language_type,
        )
        print(f"[data] loaded {len(self.items)} questions from {self.config.data_path}")

    def format_prompt(self, question_ls: list[str], language_type: str) -> list[list[dict]]:
        language = Language(language_type)
        formatted: list[list[dict]] = []
        for q in question_ls:
            msg = [
                {"role": "system", "content": language.system_prompt},
                {
                    "role": "user",
                    "content": language.user_prompt.replace("{question}", q),
                },
            ]
            formatted.append(msg)
        return formatted

    def _chat_once(self, messages: list[dict]) -> list[str]:
        """Sample n completions for one prompt; retry on transient errors."""
        last_err: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_tokens,
                    n=self.config.n,
                )
                texts = [(c.message.content or "").strip() for c in resp.choices]
                if len(texts) < self.config.n:
                    # some servers ignore n>1; pad by extra calls
                    while len(texts) < self.config.n:
                        extra = self.client.chat.completions.create(
                            model=self.config.model,
                            messages=messages,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                            max_tokens=self.config.max_tokens,
                            n=1,
                        )
                        texts.append((extra.choices[0].message.content or "").strip())
                return texts[: self.config.n]
            except Exception as e:  # noqa: BLE001
                last_err = e
                wait = min(2**attempt, 30)
                print(f"[warn] attempt {attempt}/{self.config.max_retries} failed: {e}; sleep {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"chat failed after retries: {last_err}")

    def _pick_golden(self, res_ls: list[str]) -> str:
        pick = self.config.pick
        nonempty = [r for r in res_ls if r]
        if not nonempty:
            return ""
        if pick == "first":
            return nonempty[0]
        if pick == "longest":
            return max(nonempty, key=len)
        if pick == "shortest":
            return min(nonempty, key=len)
        raise ValueError(f"unknown --pick={pick}")

    def _load_resume(self) -> dict[str, dict]:
        """Map question -> raw record for resume."""
        path = self.config.raw_save_path or self.config.save_path
        p = Path(path)
        if not (self.config.resume and p.exists()):
            return {}
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        done = {}
        for row in data:
            if "question" in row and ("res_ls" in row or "golden_res" in row or "res" in row):
                done[row["question"]] = row
        print(f"[resume] loaded {len(done)} existing rows from {p}")
        return done

    def sample_res(self) -> tuple[list[dict], list[dict]]:
        """
        Returns:
          raw_rows:  question / answer / res_ls / golden_res (+ metadata)
          sft_rows:  question / golden_res   (for example_sft.py)
        """
        done = self._load_resume()
        raw_rows: list[dict] = []
        pending_idx: list[int] = []

        for i, item in enumerate(self.items):
            q = item["question"]
            if q in done:
                prev = done[q]
                if "res_ls" in prev:
                    res_ls = prev["res_ls"]
                elif "res" in prev:
                    res_ls = [prev["res"]]
                else:
                    res_ls = [prev.get("golden_res", "")]
                golden = prev.get("golden_res") or self._pick_golden(res_ls)
                raw_rows.append(
                    {
                        **item,
                        "res_ls": res_ls,
                        "golden_res": golden,
                        "model": prev.get("model", self.config.model),
                        "language_type": prev.get("language_type", self.config.language_type),
                    }
                )
            else:
                pending_idx.append(i)
                raw_rows.append({})  # placeholder

        print(f"[sample] total={len(self.items)} resume={len(self.items) - len(pending_idx)} todo={len(pending_idx)}")

        def _job(i: int) -> tuple[int, list[str]]:
            return i, self._chat_once(self.messages_ls[i])

        if pending_idx:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as ex:
                futs = [ex.submit(_job, i) for i in pending_idx]
                for fut in tqdm(as_completed(futs), total=len(futs), desc="sample_res"):
                    i, res_ls = fut.result()
                    item = self.items[i]
                    golden = self._pick_golden(res_ls)
                    raw_rows[i] = {
                        **item,
                        "res_ls": res_ls,
                        "golden_res": golden,
                        "model": self.config.model,
                        "language_type": self.config.language_type,
                    }

        # drop any hole (should not happen)
        raw_rows = [r for r in raw_rows if r]
        sft_rows = [
            {"question": r["question"], "golden_res": r["golden_res"]}
            for r in raw_rows
            if r.get("golden_res")
        ]
        skipped = len(raw_rows) - len(sft_rows)
        if skipped:
            print(f"[warn] skipped {skipped} empty golden_res rows in SFT export")
        return raw_rows, sft_rows

    def save(self, raw_rows: list[dict], sft_rows: list[dict]):
        save_path = Path(self.config.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(sft_rows, f, ensure_ascii=False, indent=2)
        print(f"[save] SFT data ({len(sft_rows)}) -> {save_path}")

        if self.config.raw_save_path:
            raw_path = Path(self.config.raw_save_path)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(raw_rows, f, ensure_ascii=False, indent=2)
            print(f"[save] raw samples ({len(raw_rows)}) -> {raw_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample teacher responses for SFT distillation")
    p.add_argument("--base_url", type=str, required=True, help="OpenAI-compatible base url, e.g. http://127.0.0.1:8000/v1")
    p.add_argument("--api_key", type=str, default="EMPTY")
    p.add_argument("--model", type=str, required=True, help="teacher model name served by the API")
    p.add_argument(
        "--data_path",
        type=str,
        default="/root/hidden_prob/data/math/train_split3000.json",
    )
    p.add_argument(
        "--save_path",
        type=str,
        default="/root/autodl-tmp/exp3_sft/teacher/math_en_sft.json",
        help="SFT json: list[{question,golden_res}]",
    )
    p.add_argument(
        "--raw_save_path",
        type=str,
        default=None,
        help="optional raw dump with res_ls / answer / metadata",
    )
    p.add_argument("--language_type", type=str, default="en", choices=["en", "zh", "es", "vi", "tr"])
    p.add_argument("--n", type=int, default=1, help="samples per question")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--max_workers", type=int, default=8, help="concurrent API requests")
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument(
        "--pick",
        type=str,
        default="first",
        choices=["first", "longest", "shortest"],
        help="how to choose golden_res when n>1",
    )
    p.add_argument("--limit", type=int, default=None, help="only first N questions (debug)")
    p.add_argument("--resume", action="store_true", help="skip questions already in raw/save json")
    return p.parse_args()


def main():
    args = parse_args()
    config = Config(
        data_path=args.data_path,
        save_path=args.save_path,
        raw_save_path=args.raw_save_path,
        language_type=args.language_type,
        model=args.model,
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        pick=args.pick,
        limit=args.limit,
        resume=args.resume,
    )
    client = Client(base_url=args.base_url, api_key=args.api_key, config=config)
    raw_rows, sft_rows = client.sample_res()
    client.save(raw_rows, sft_rows)
    print("[done]")


if __name__ == "__main__":
    main()

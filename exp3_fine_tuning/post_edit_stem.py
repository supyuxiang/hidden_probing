"""
STEM hard-sentence LLM post-editing (Nemotron-style).

After English CoT is translated into a target language, many STEM samples break
LaTeX / \\boxed{} / step formatting / language consistency. This script asks an
editor LLM to *fix the translation* (not re-solve from scratch), then applies
heuristic filters before writing SFT-ready JSON.

python /root/hidden_prob/exp3_fine_tuning/post_edit_stem.py \
    --base_url http://127.0.0.1:8000/v1 \
    --api_key EMPTY \
    --model Qwen2.5-14B-Instruct \
    --data_path /root/autodl-tmp/exp3_sft/teacher/math_es_translated.json \
    --save_path /root/autodl-tmp/exp3_sft/teacher/math_es_postedited.json \
    --raw_save_path /root/autodl-tmp/exp3_sft/teacher/math_es_postedited_raw.json \
    --target_lang Spanish \
    --require_boxed \
    --drop_failed \
    --only_hard
    
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

BOXED_RE = re.compile(r"\\boxed\s*\{")
LATEX_HINT_RE = re.compile(r"(\$.*?\$|\\frac|\\sqrt|\\sum|\\int|\\begin\{)")


EDITOR_SYSTEM = (
    "You are a careful STEM translation post-editor. "
    "You fix formatting and translation errors in already-translated math/science/"
    "coding solutions. You do NOT invent a new solution unless the text is "
    "clearly corrupted beyond repair."
)


def build_editor_prompt(
    target_lang: str,
    question: str,
    solution: str,
    en_question: str | None,
    en_solution: str | None,
    gold_answer: str | None,
) -> str:
    parts = [
        f"Target language: {target_lang}.",
        "Task: post-edit the TRANSLATED STEM sample.",
        "Goals:",
        "1) Keep mathematical meaning identical to the source reasoning.",
        "2) Repair broken LaTeX, markdown, enumerations, and \\boxed{...}.",
        "3) Ensure the full solution (and final answer) is written in the target language,",
        "   but keep math tokens/LaTeX/identifiers unchanged.",
        "4) Do not add new theorems or change the final answer unless the current boxed",
        "   answer is malformed; if a gold answer is provided, the boxed result must match it.",
        "5) If the translation is already good, return it with only minimal fixes.",
        "",
        "Return ONLY the post-edited solution text. No preamble.",
        "",
        f"[Translated question]\n{question}",
        "",
        f"[Translated solution to edit]\n{solution}",
    ]
    if en_question:
        parts.extend(["", f"[English question (reference)]\n{en_question}"])
    if en_solution:
        parts.extend(["", f"[English solution (reference)]\n{en_solution}"])
    if gold_answer:
        parts.extend(["", f"[Gold final answer]\n{gold_answer}"])
    return "\n".join(parts)


def _first(row: dict, keys: list[str], default: str = "") -> str:
    for k in keys:
        if k in row and row[k] is not None and str(row[k]).strip():
            return str(row[k])
    return default


def normalize_row(row: dict) -> dict:
    question = _first(row, ["question", "problem", "prompt", "query"])
    solution = _first(
        row,
        ["golden_res", "solution", "response", "answer_text", "output", "res"],
    )
    if not question or not solution:
        raise KeyError(f"missing question/solution fields; keys={list(row.keys())}")
    return {
        "question": question,
        "golden_res": solution,
        "answer": _first(row, ["answer", "gold_answer", "final_answer"]),
        "en_question": _first(row, ["en_question", "question_en", "src_question"]),
        "en_golden_res": _first(
            row, ["en_golden_res", "en_solution", "solution_en", "src_solution"]
        ),
        "meta": {k: v for k, v in row.items() if k not in {
            "question", "problem", "prompt", "query",
            "golden_res", "solution", "response", "answer_text", "output", "res",
            "answer", "gold_answer", "final_answer",
            "en_question", "question_en", "src_question",
            "en_golden_res", "en_solution", "solution_en", "src_solution",
        }},
    }


def needs_post_edit(solution: str, require_boxed: bool) -> tuple[bool, list[str]]:
    """Heuristic: only send hard/broken STEM samples to the editor if --only_hard."""
    reasons: list[str] = []
    if require_boxed and not BOXED_RE.search(solution):
        reasons.append("missing_boxed")
    # unbalanced $ or { } are common MT failure modes
    if solution.count("$") % 2 == 1:
        reasons.append("odd_dollar")
    if solution.count("{") != solution.count("}"):
        reasons.append("unbalanced_braces")
    if "\\boxed" in solution and not BOXED_RE.search(solution):
        reasons.append("broken_boxed_cmd")
    # translated latex commands sometimes get spaces / missing backslash
    if re.search(r"\bfrac\s*\{", solution) and "\\frac" not in solution:
        reasons.append("missing_backslash_frac")
    if LATEX_HINT_RE.search(solution) and len(solution) < 30:
        reasons.append("too_short_with_latex")
    return (len(reasons) > 0), reasons


def passes_filters(
    question: str,
    solution: str,
    gold_answer: str | None,
    require_boxed: bool,
    min_chars: int,
) -> tuple[bool, list[str]]:
    fails: list[str] = []
    if len(solution.strip()) < min_chars:
        fails.append("too_short")
    if require_boxed and not BOXED_RE.search(solution):
        fails.append("missing_boxed")
    if solution.count("{") != solution.count("}"):
        fails.append("unbalanced_braces")
    if solution.count("$") % 2 == 1:
        fails.append("odd_dollar")
    if gold_answer:
        # soft check: gold answer string appears near the end / in boxed region
        ga = gold_answer.strip()
        if ga and ga not in solution and ga.replace(" ", "") not in solution.replace(" ", ""):
            fails.append("gold_answer_not_found")
    if not question.strip():
        fails.append("empty_question")
    return (len(fails) == 0), fails


@dataclass
class Config:
    data_path: str
    save_path: str
    raw_save_path: str | None
    model: str
    target_lang: str
    temperature: float
    top_p: float
    max_tokens: int
    max_workers: int
    max_retries: int
    only_hard: bool
    require_boxed: bool
    drop_failed: bool
    min_chars: int
    limit: int | None
    resume: bool


class StemPostEditor:
    def __init__(self, base_url: str, api_key: str, config: Config):
        self.config = config
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.rows = self._load()

    def _load(self) -> list[dict]:
        with open(self.config.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list) and data, f"empty data: {self.config.data_path}"
        rows = [normalize_row(x) for x in data]
        if self.config.limit is not None:
            rows = rows[: self.config.limit]
        print(f"[data] loaded {len(rows)} rows from {self.config.data_path}")
        return rows

    def _chat(self, content: str) -> str:
        last_err: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": EDITOR_SYSTEM},
                        {"role": "user", "content": content},
                    ],
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_tokens,
                    n=1,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                wait = min(2**attempt, 30)
                print(f"[warn] attempt {attempt}/{self.config.max_retries} failed: {e}; sleep {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"editor chat failed: {last_err}")

    def _load_resume(self) -> dict[str, dict]:
        path = self.config.raw_save_path
        if not (self.config.resume and path and Path(path).exists()):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        done = {r["question"]: r for r in data if "question" in r and "golden_res" in r}
        print(f"[resume] {len(done)} rows from {path}")
        return done

    def run(self) -> tuple[list[dict], list[dict]]:
        done = self._load_resume()
        raw_out: list[dict] = [None] * len(self.rows)  # type: ignore
        todo: list[int] = []

        for i, row in enumerate(self.rows):
            q = row["question"]
            if q in done:
                raw_out[i] = done[q]
            else:
                todo.append(i)

        print(
            f"[edit] total={len(self.rows)} resume={len(self.rows) - len(todo)} todo={len(todo)} "
            f"only_hard={self.config.only_hard}"
        )

        def job(i: int) -> tuple[int, dict]:
            row = self.rows[i]
            before = row["golden_res"]
            hard, hard_reasons = needs_post_edit(before, self.config.require_boxed)
            edited = before
            did_edit = False
            if (not self.config.only_hard) or hard:
                prompt = build_editor_prompt(
                    target_lang=self.config.target_lang,
                    question=row["question"],
                    solution=before,
                    en_question=row["en_question"] or None,
                    en_solution=row["en_golden_res"] or None,
                    gold_answer=row["answer"] or None,
                )
                edited = self._chat(prompt)
                did_edit = True

            ok, fail_reasons = passes_filters(
                question=row["question"],
                solution=edited,
                gold_answer=row["answer"] or None,
                require_boxed=self.config.require_boxed,
                min_chars=self.config.min_chars,
            )
            rec = {
                **row.get("meta", {}),
                "question": row["question"],
                "golden_res_before": before,
                "golden_res": edited,
                "answer": row["answer"],
                "en_question": row["en_question"],
                "en_golden_res": row["en_golden_res"],
                "did_edit": did_edit,
                "hard": hard,
                "hard_reasons": hard_reasons,
                "pass": ok,
                "fail_reasons": fail_reasons,
                "editor_model": self.config.model,
                "target_lang": self.config.target_lang,
            }
            return i, rec

        if todo:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as ex:
                futs = [ex.submit(job, i) for i in todo]
                for fut in tqdm(as_completed(futs), total=len(futs), desc="post_edit"):
                    i, rec = fut.result()
                    raw_out[i] = rec

        raw_rows = [r for r in raw_out if r is not None]
        if self.config.drop_failed:
            kept = [r for r in raw_rows if r.get("pass", False)]
            print(f"[filter] keep {len(kept)}/{len(raw_rows)} passing rows")
        else:
            kept = raw_rows

        sft_rows = [{"question": r["question"], "golden_res": r["golden_res"]} for r in kept]
        return raw_rows, sft_rows

    def save(self, raw_rows: list[dict], sft_rows: list[dict]):
        save_path = Path(self.config.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(sft_rows, f, ensure_ascii=False, indent=2)
        print(f"[save] SFT ({len(sft_rows)}) -> {save_path}")

        if self.config.raw_save_path:
            raw_path = Path(self.config.raw_save_path)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(raw_rows, f, ensure_ascii=False, indent=2)
            print(f"[save] raw ({len(raw_rows)}) -> {raw_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STEM translation post-editing with an LLM")
    p.add_argument("--base_url", type=str, required=True)
    p.add_argument("--api_key", type=str, default="EMPTY")
    p.add_argument("--model", type=str, required=True, help="editor model (14B/32B instruct is enough)")
    p.add_argument("--data_path", type=str, required=True, help="translated STEM json list")
    p.add_argument("--save_path", type=str, required=True, help="SFT json list[{question,golden_res}]")
    p.add_argument("--raw_save_path", type=str, default=None)
    p.add_argument("--target_lang", type=str, default="Spanish", help="e.g. Spanish / Turkish / Chinese")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--max_workers", type=int, default=8)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument(
        "--only_hard",
        action="store_true",
        help="only call LLM when heuristics detect broken STEM formatting",
    )
    p.add_argument("--require_boxed", action="store_true", help="require \\boxed{...} after editing")
    p.add_argument("--drop_failed", action="store_true", help="drop rows failing heuristic filters")
    p.add_argument("--min_chars", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    config = Config(
        data_path=args.data_path,
        save_path=args.save_path,
        raw_save_path=args.raw_save_path,
        model=args.model,
        target_lang=args.target_lang,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        only_hard=args.only_hard,
        require_boxed=args.require_boxed,
        drop_failed=args.drop_failed,
        min_chars=args.min_chars,
        limit=args.limit,
        resume=args.resume,
    )
    editor = StemPostEditor(args.base_url, args.api_key, config)
    raw_rows, sft_rows = editor.run()
    editor.save(raw_rows, sft_rows)
    print("[done]")


if __name__ == "__main__":
    main()

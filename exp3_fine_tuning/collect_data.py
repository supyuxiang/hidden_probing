"""
Offline teacher sampling with local vLLM for later best-of-n + SFT distillation.

Default teacher: DeepSeek-R1-Distill-Qwen-32B
Each question gets n samples (default 4). Keep all candidates in `res_ls` for an
external reward model to pick the best later.

Example:
  CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp3_fine_tuning/collect_data.py \
      --model_path /root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-32B \
      --data_path /root/hidden_prob/data/math/train_split3000.json \
      --save_path /root/autodl-tmp/exp3_sft/teacher/math_en_n4_candidates.json \
      --language_type en \
      --n 4 \
      --temperature 0.6 \
      --max_tokens 1024 \
      --gpu_memory_utilization 0.9

Optional preliminary SFT export (before RM; uses --pick):
  ... --sft_save_path /root/autodl-tmp/exp3_sft/teacher/math_en_sft_tmp.json --pick first
"""



import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exp1_math.language import Language  # noqa: E402

DEFAULT_MODEL = "/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-32B"


def load_items(data_path: str | Path, limit: int | None) -> list[dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and data, f"empty data: {data_path}"

    items: list[dict] = []
    for i, row in enumerate(data):
        if "question" not in row:
            raise KeyError(f"row {i} missing 'question'; keys={list(row.keys())}")
        item = {
            "question": row["question"],
            "answer": row.get("answer", ""),
        }
        for k in ("subject", "level", "solution"):
            if k in row:
                item[k] = row[k]
        items.append(item)

    if limit is not None:
        items = items[:limit]
    print(f"[data] loaded {len(items)} questions from {data_path}")
    return items


def format_prompts(
    question_ls: list[str],
    tokenizer: AutoTokenizer,
    language_type: str,
) -> list[str]:
    """
    DeepSeek-R1-Distill recommends putting instructions in the user message.
    We still use chat template; system prompt is kept light via Language.
    """
    language = Language(language_type)
    prompts: list[str] = []
    for q in question_ls:
        # Prefer user-side instruction for R1-distill stability.
        user = language.user_prompt.replace("{question}", q)
        msg = [
            {"role": "system", "content": language.system_prompt},
            {"role": "user", "content": user},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def pick_one(res_ls: list[str], pick: str) -> str:
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


def load_resume(path: str | Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    done = {}
    for row in data:
        if "question" in row and "res_ls" in row and isinstance(row["res_ls"], list):
            done[row["question"]] = row
    print(f"[resume] loaded {len(done)} finished questions from {p}")
    return done


def sample_with_vllm(
    items: list[dict],
    prompts: list[str],
    llm: LLM,
    sp: SamplingParams,
    model_path: str,
    language_type: str,
    resume_map: dict[str, dict],
) -> list[dict]:
    todo_idx = [i for i, it in enumerate(items) if it["question"] not in resume_map]
    print(
        f"[sample] total={len(items)} resume={len(items) - len(todo_idx)} todo={len(todo_idx)} n={sp.n}"
    )

    out: list[dict] = [None] * len(items)  # type: ignore
    for i, it in enumerate(items):
        q = it["question"]
        if q in resume_map:
            prev = resume_map[q]
            out[i] = {
                **it,
                "res_ls": prev["res_ls"],
                "model": prev.get("model", model_path),
                "language_type": prev.get("language_type", language_type),
            }

    if todo_idx:
        todo_prompts = [prompts[i] for i in todo_idx]
        outputs = llm.generate(todo_prompts, sp)
        for local_j, i in enumerate(tqdm(todo_idx, desc="pack")):
            res_ls = [(o.text or "").strip() for o in outputs[local_j].outputs]
            # pad if server returned fewer than n (should not happen with vLLM)
            while len(res_ls) < sp.n:
                res_ls.append("")
            out[i] = {
                **items[i],
                "res_ls": res_ls[: sp.n],
                "model": model_path,
                "language_type": language_type,
            }

    return [r for r in out if r is not None]


def save_json(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[save] {len(obj) if isinstance(obj, list) else 'ok'} -> {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="vLLM offline sampling (n candidates / question) for RM best-of-n + SFT"
    )
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    p.add_argument(
        "--data_path",
        type=str,
        default="/root/hidden_prob/data/math/train_split3000.json",
    )
    p.add_argument(
        "--save_path",
        type=str,
        default="/root/autodl-tmp/exp3_sft/teacher/math_en_n4_candidates.json",
        help="candidates json: list[{question,answer,res_ls,...}]",
    )
    p.add_argument(
        "--sft_save_path",
        type=str,
        default=None,
        help="optional preliminary SFT export list[{question,golden_res}] via --pick",
    )
    p.add_argument("--language_type", type=str, default="en", choices=["en", "zh", "es", "vi", "tr"])
    p.add_argument("--n", type=int, default=4, help="samples per question (for later best-of-n)")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=None, help="optional vLLM context cap")
    p.add_argument("--dtype", type=str, default="auto", help="auto|float16|bfloat16")
    p.add_argument("--limit", type=int, default=None, help="only first N questions (debug)")
    p.add_argument("--resume", action="store_true", help="skip questions already in --save_path")
    p.add_argument(
        "--pick",
        type=str,
        default="first",
        choices=["first", "longest", "shortest"],
        help="only used when --sft_save_path is set (before external RM)",
    )
    p.add_argument(
        "--flatten_save_path",
        type=str,
        default=None,
        help="optional flat dump list[{question,answer,res,sample_id}] like exp1 sample_res",
    )
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import set_seed as hf_set_seed

    hf_set_seed(args.seed)

    items = load_items(args.data_path, args.limit)
    resume_map = load_resume(args.save_path) if args.resume else {}

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = format_prompts(
        [x["question"] for x in items],
        tokenizer,
        args.language_type,
    )

    llm_kwargs = dict(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype=args.dtype,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    print(f"[vllm] loading {args.model_path} ...")
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.n,
        seed=args.seed,
    )

    rows = sample_with_vllm(
        items=items,
        prompts=prompts,
        llm=llm,
        sp=sp,
        model_path=args.model_path,
        language_type=args.language_type,
        resume_map=resume_map,
    )
    save_json(rows, args.save_path)

    if args.flatten_save_path:
        flat = []
        for row in rows:
            for sid, text in enumerate(row["res_ls"]):
                flat.append(
                    {
                        "question": row["question"],
                        "answer": row.get("answer", ""),
                        "res": text,
                        "sample_id": sid,
                        "model": row.get("model", args.model_path),
                        "language_type": row.get("language_type", args.language_type),
                    }
                )
        save_json(flat, args.flatten_save_path)

    if args.sft_save_path:
        sft_rows = []
        for row in rows:
            golden = pick_one(row["res_ls"], args.pick)
            if golden:
                sft_rows.append({"question": row["question"], "golden_res": golden})
        save_json(sft_rows, args.sft_save_path)
        print(f"[note] SFT export used --pick={args.pick}; replace after external RM best-of-n")

    print("[done]")


if __name__ == "__main__":
    main()

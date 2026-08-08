"""
Greedy baseline: 4 models × 5 multilingual math test sets.
Saves rollouts + judge scores (acc).

Example:
  CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp3_fine_tuning/baseline.py \
      --out_dir /root/autodl-tmp/exp3_sft/baseline_greedy \
      --judge_model /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
      --gpu_memory_utilization 0.95

"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exp1_math.language import Language
from exp1_math.judge import judge_math

DEFAULT_MODELS = [
    "/root/autodl-tmp/models/Qwen2.5-3B-Instruct",
    "/root/autodl-tmp/models/Qwen2.5-14B-Instruct",
    "/root/autodl-tmp/models/Qwen3-8B",
    "/root/autodl-tmp/models/Llama3.2-8B-Instruct",
]

DEFAULT_LANGS = ["en", "zh", "es", "vi", "tr"]

TEST_DATA = {
    "en": "/root/hidden_prob/data/math/test.json",
    "zh": "/root/hidden_prob/data/math/test_zh.json",
    "es": "/root/hidden_prob/data/math/test_es.json",
    "vi": "/root/hidden_prob/data/math/test_vi.json",
    "tr": "/root/hidden_prob/data/math/test_tr.json",
}


def load_qa(data_path: str | Path, limit: int | None) -> tuple[list[str], list[str]]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and data, f"empty data: {data_path}"
    if limit is not None:
        data = data[:limit]
    return [x["question"] for x in data], [x["answer"] for x in data]


def format_prompts(
    question_ls: list[str],
    tokenizer: AutoTokenizer,
    language_type: str,
) -> list[str]:
    language = Language(language_type)
    out: list[str] = []
    for q in question_ls:
        msg = [
            {"role": "system", "content": language.system_prompt},
            {
                "role": "user",
                "content": language.user_prompt.replace("{question}", q),
            },
        ]
        out.append(
            tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
        )
    return out


def greedy_rollout(
    question_ls: list[str],
    answer_ls: list[str],
    tokenizer: AutoTokenizer,
    llm: LLM,
    language_type: str,
    max_tokens: int,
    seed: int,
) -> list[dict]:
    prompts = format_prompts(question_ls, tokenizer, language_type)
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        n=1,
        seed=seed,
    )
    outputs = llm.generate(prompts, sp)
    rows: list[dict] = []
    for q, a, o in zip(question_ls, answer_ls, outputs):
        rows.append(
            {
                "question": q,
                "answer": a,
                "res": o.outputs[0].text,
            }
        )
    return rows


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Greedy baseline on multilingual math test sets")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--langs", nargs="+", default=DEFAULT_LANGS)
    p.add_argument(
        "--out_dir",
        type=str,
        default="/root/autodl-tmp/exp3_sft/baseline_greedy",
    )
    p.add_argument(
        "--judge_model",
        type=str,
        default="/root/autodl-tmp/models/Qwen2.5-14B-Instruct",
    )
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--judge_max_model_len", type=int, default=20000)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="only first N questions per lang")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- phase 1: greedy rollouts ----------
    for model_path in args.models:
        model_name = Path(model_path).name
        model_out = out_dir / model_name
        model_out.mkdir(parents=True, exist_ok=True)

        print(f"\n===== generate | {model_name} | max_tokens={args.max_tokens} | max_model_len={args.max_model_len} | langs={args.langs} =====")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        llm = LLM(
            model_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
        )

        for lang in args.langs:
            data_path = TEST_DATA[lang]
            question_ls, answer_ls = load_qa(data_path, args.limit)
            rows = greedy_rollout(
                question_ls=question_ls,
                answer_ls=answer_ls,
                tokenizer=tokenizer,
                llm=llm,
                language_type=lang,
                max_tokens=args.max_tokens,
                seed=args.seed,
            )
            rollout_path = model_out / f"rollout_test_{lang}_greedy_tokens{args.max_tokens}.json"
            save_json(rows, rollout_path)

        del llm, tokenizer, rows
        gc.collect()
        torch.cuda.empty_cache()

    # ---------- phase 2: judge ----------
    print(f"\n===== judge | {args.judge_model} =====")
    judge_tok = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
    if judge_tok.pad_token is None:
        judge_tok.pad_token = judge_tok.eos_token
    judge_llm = LLM(
        args.judge_model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.judge_max_model_len,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    summary_rows: list[dict] = []
    for model_path in args.models:
        model_name = Path(model_path).name
        model_out = out_dir / model_name
        tokens = args.max_tokens

        for lang in args.langs:
            rollout_path = model_out / f"rollout_test_{lang}_greedy_tokens{tokens}.json"
            assert rollout_path.exists(), f"missing rollout: {rollout_path}"
            with open(rollout_path, "r", encoding="utf-8") as f:
                rows = json.load(f)

            res_ls = [r["res"] for r in rows]
            answer_ls = [r["answer"] for r in rows]
            
            judge_bools, rewards = judge_math(
                res_ls=res_ls,
                answer_ls=answer_ls,
                llm=judge_llm,
                tokenizer=judge_tok,
            )
            acc = float(rewards.float().mean().item())

            for r, ok, reward in zip(rows, judge_bools, rewards.view(-1).tolist()):
                r["correct"] = bool(ok)
                r["reward"] = float(reward)

            scored_path = model_out / f"scored_test_{lang}_greedy_tokens{tokens}.json"
            reward_path = model_out / f"reward_test_{lang}_greedy_tokens{tokens}.pt"
            save_json(rows, scored_path)
            torch.save(rewards.detach().cpu(), reward_path)
            print(f"[save] {reward_path} | acc={acc:.4f}")

            summary_rows.append(
                {
                    "model": model_name,
                    "lang": lang,
                    "n": len(rows),
                    "max_tokens": tokens,
                    "acc": acc,
                    "rollout": str(rollout_path),
                    "scored": str(scored_path),
                }
            )

    del judge_llm, judge_tok
    gc.collect()
    torch.cuda.empty_cache()

    summary_path = out_dir / "summary.json"
    save_json(summary_rows, summary_path)

    langs = args.langs
    models = [Path(m).name for m in args.models]
    print("\n===== greedy ACC =====")
    header = f"{'model':<28}" + "".join(f"{l:>10}" for l in langs)
    print(header)
    print("-" * len(header))
    for model_name in models:
        cells = []
        for lang in langs:
            hit = next(
                (r for r in summary_rows if r["model"] == model_name and r["lang"] == lang),
                None,
            )
            cells.append(f"{hit['acc']*100:5.1f}" if hit else "   —")
        print(f"{model_name:<28}" + "".join(f"{c:>10}" for c in cells))
    print(f"\nsummary -> {summary_path}")
    print("all done!")


if __name__ == "__main__":
    main()

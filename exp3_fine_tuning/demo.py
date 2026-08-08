"""
Quick demo: translate first N SFT samples and print output format.

Example:
  CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp3_fine_tuning/demo.py \
      --tgt_lang es \
      --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from translate_sft_data import (  # noqa: E402
    DEFAULT_DATA_PATH,
    DEFAULT_TRANSLATOR,
    LANG_NAME,
    format_prompts,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo translate first N SFT rows")
    p.add_argument("--model_path", type=str, default=DEFAULT_TRANSLATOR)
    p.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    p.add_argument("--tgt_lang", type=str, default="es", choices=[c for c in LANG_NAME if c != "en"])
    p.add_argument("--src_lang", type=str, default="en")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--preview_chars", type=int, default=400, help="truncate printed text")
    return p.parse_args()


def preview(text: str, n: int) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= n else text[:n] + f"...(+{len(text) - n} chars)"


def main() -> None:
    args = parse_args()

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = data[: args.limit]
    print(f"[data] loaded {len(data)} rows from {args.data_path}")
    print(f"[lang] {args.src_lang} -> {args.tgt_lang}")

    questions = [str(x["question"]) for x in data]
    answers = [str(x["answer"]) for x in data]
    golden_res = [str(x["golden_res"]) for x in data]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # translate question + golden_res for each row
    q_prompts = format_prompts(questions, tokenizer, args.src_lang, args.tgt_lang)
    r_prompts = format_prompts(golden_res, tokenizer, args.src_lang, args.tgt_lang)

    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        top_k=50,
        max_tokens=args.max_tokens,
        n=1,
        seed=42,
    )

    print("[gen] translating questions...")
    q_out = [o.outputs[0].text.strip() for o in llm.generate(q_prompts, sp)]
    print("[gen] translating golden_res...")
    r_out = [o.outputs[0].text.strip() for o in llm.generate(r_prompts, sp)]

    rows = []
    for i, (q, a, en_q, en_r, tq, tr) in enumerate(
        zip(questions, answers, questions, golden_res, q_out, r_out)
    ):
        item = {
            "question": tq,
            "golden_res": tr,
            "answer": a,
            "en_question": en_q,
            "en_golden_res": en_r,
            "src_lang": args.src_lang,
            "tgt_lang": args.tgt_lang,
        }
        rows.append(item)

        print("=" * 80)
        print(f"[{i}] keys={list(item.keys())}")
        print(f"answer (unchanged): {a}")
        print(f"en_question: {preview(en_q, args.preview_chars)}")
        print(f"question   : {preview(tq, args.preview_chars)}")
        print(f"en_golden_res: {preview(en_r, args.preview_chars)}")
        print(f"golden_res   : {preview(tr, args.preview_chars)}")

    print("=" * 80)
    print("[format] example JSON row:")
    print(json.dumps(rows[0], ensure_ascii=False, indent=2)[:2000])
    print("... (truncated if long)")
    print(f"[done] translated {len(rows)} rows")


if __name__ == "__main__":
    main()

"""
Translate sampled teacher SFT / candidate data into another language.

Typical pipeline:
  1) collect_data.py  -> English candidates {question, answer, res_ls}
  2) translate_sft_data.py (this) -> target-lang question + res_ls
  3) post_edit_stem.py (optional) -> fix STEM formatting
  4) external RM best-of-n -> SFT {question, golden_res}

Keeps English source fields for reference:
  en_question, en_res_ls / en_golden_res
Final answer field `answer` is kept unchanged (math gold).

Example:
  CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp3_fine_tuning/translate_sft_data.py \
      --model_path /root/autodl-tmp/models/Qwen2.5-32B-Instruct \
      --data_path /root/hidden_prob/exp3_fine_tuning/teacher/math_en_n2_sft.json \
      --save_path /root/hidden_prob/exp3_fine_tuning/teacher/math_es_n2_sft_translated.json \
      --src_lang en \
      --tgt_lang es \
      --temperature 0.2 \
      --max_tokens 4096 \
      --max_model_len 8192 \
      --gpu_memory_utilization 0.9
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

LANG_NAME = {
    "en": "English",
    "zh": "Simplified Chinese",
    "es": "Spanish",
    "vi": "Vietnamese",
    "tr": "Turkish",
}

DEFAULT_TRANSLATOR = "/root/autodl-tmp/models/Qwen2.5-32B-Instruct"
DEFAULT_DATA_PATH = '/root/hidden_prob/exp3_fine_tuning/teacher/math_en_n2_sft.json'
DEFAULT_SAVE_DIR = Path("/root/hidden_prob/exp3_fine_tuning/teacher")

TRANSLATE_SYSTEM = (
    "You are a professional translator. "
    "Translate faithfully. Preserve all LaTeX, math expressions, code, "
    "variable names, \\boxed{...}, and markdown structure. "
    "Do not solve the problem or add new content. "
    "Output ONLY the translation."
)



def load_rows(data_path: str | Path, limit: int | None) -> list[dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and data, f"empty data: {data_path}"
    # question, answer, golden_res

    rows: list[dict] = []
    for i, row in enumerate(data):
        rows.append(
            {
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                'golden_res': str(row['golden_res'])
            }
        )
        
    if limit is not None:
        rows = rows[:limit]
    return rows


def build_translate_prompt(
    text: str,
    tokenizer: AutoTokenizer,
    src_lang: str,
    tgt_lang: str,
    text_type: str,
) -> str:
    user = (
        f"Translate the following {text_type} from {LANG_NAME(src_lang)} "
        f"to {LANG_NAME(tgt_lang)}.\n"
        "Requirements:\n"
        f"- Write the translation in {LANG_NAME(tgt_lang)}.\n"
        "- Keep LaTeX / formulas / \\boxed{{...}} / code unchanged in structure.\n"
        "- Do not answer the question; only translate.\n"
        "- Output the translation only.\n\n"
        f"{text_type.capitalize()}:\n{text}"
    )
    msg = [
        {"role": "system", "content": TRANSLATE_SYSTEM},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)



def flatten_jobs(rows: list[dict], translate_question: bool) -> list[dict]:
    """Each job: {row_idx, field: 'question'|'sol', sol_idx?, src_text}."""
    jobs: list[dict] = []
    for i, row in enumerate(rows):
        if translate_question:
            jobs.append(
                {
                    "row_idx": i,
                    "field": "question",
                    "sol_idx": None,
                    "src_text": row["question"],
                    "text_type": "math question",
                }
            )
        for j, sol in enumerate(row["texts"]):
            jobs.append(
                {
                    "row_idx": i,
                    "field": "sol",
                    "sol_idx": j,
                    "src_text": sol,
                    "text_type": "math solution",
                }
            )
    return jobs


def pack_outputs(
    rows: list[dict],
    jobs: list[dict],
    translations: list[str],
    src_lang: str,
    tgt_lang: str,
    model_path: str,
    translate_question: bool,
) -> list[dict]:
    # defaults: copy source if somehow missing
    q_tr = [r["question"] for r in rows]
    sols_tr: list[list[str]] = [list(r["texts"]) for r in rows]

    for job, tr in zip(jobs, translations):
        i = job["row_idx"]
        text = (tr or "").strip()
        if job["field"] == "question":
            q_tr[i] = text if text else rows[i]["question"]
        else:
            j = job["sol_idx"]
            sols_tr[i][j] = text if text else rows[i]["texts"][j]

    out: list[dict] = []
    for i, row in enumerate(rows):
        item = {
            **row.get("meta", {}),
            "question": q_tr[i] if translate_question else row["question"],
            "answer": row["answer"],  # keep gold answer as-is
            "en_question": row["question"],
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "translator_model": model_path,
        }
        if row["kind"] == "candidates":
            item["res_ls"] = sols_tr[i]
            item["en_res_ls"] = row["texts"]
        elif row["kind"] == "sft":
            item["golden_res"] = sols_tr[i][0]
            item["en_golden_res"] = row["texts"][0]
        else:  # flat
            item["res"] = sols_tr[i][0]
            item["en_res"] = row["texts"][0]
            if row["sample_id"] is not None:
                item["sample_id"] = row["sample_id"]
        out.append(item)
    return out


def maybe_export_sft(rows: list[dict], path: str | None, pick: str):
    if not path:
        return
    sft = []
    for r in rows:
        if "golden_res" in r:
            golden = r["golden_res"]
        elif "res_ls" in r:
            cands = [x for x in r["res_ls"] if x]
            if not cands:
                continue
            if pick == "first":
                golden = cands[0]
            elif pick == "longest":
                golden = max(cands, key=len)
            elif pick == "shortest":
                golden = min(cands, key=len)
            else:
                raise ValueError(pick)
        elif "res" in r:
            golden = r["res"]
        else:
            continue
        if golden:
            sft.append({"question": r["question"], "golden_res": golden})
    path_p = Path(path)
    path_p.parent.mkdir(parents=True, exist_ok=True)
    with open(path_p, "w", encoding="utf-8") as f:
        json.dump(sft, f, ensure_ascii=False, indent=2)
    print(f"[save] SFT export ({len(sft)}) -> {path_p}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Translate teacher SFT/candidate data with vLLM")
    p.add_argument("--model_path", type=str, default=DEFAULT_TRANSLATOR)
    p.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="EN SFT/candidate json (question + golden_res or res_ls)",
    )
    p.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="translated json path; default: teacher/math_{tgt_lang}_n2_sft_translated.json",
    )
    p.add_argument("--sft_save_path", type=str, default=None, help="optional SFT export after translate")
    p.add_argument("--src_lang", type=str, default="en", choices=list(LANG_NAME))
    p.add_argument("--tgt_lang", type=str, required=True, choices=[c for c in LANG_NAME if c != "en"])
    p.add_argument("--translate_question", action="store_true", default=True)
    p.add_argument("--no_translate_question", action="store_false", dest="translate_question")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="vLLM context cap; keep modest to fit KV cache",
    )
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--pick", type=str, default="first", choices=["first", "longest", "shortest"])
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import set_seed as hf_set_seed
    hf_set_seed(args.seed)

    if not args.save_path:
        args.save_path = str(DEFAULT_SAVE_DIR / f"math_{tgt_lang}_n2_sft_translated.json")

    rows = load_rows(args.data_path, args.limit)
    jobs = flatten_jobs(rows, translate_question=args.translate_question)
    print(
        f"[jobs] rows={len(rows)} translate_units={len(jobs)} "
        f"{args.src_lang} -> {args.tgt_lang}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        build_translate_prompt(
            text=j["src_text"],
            tokenizer=tokenizer,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            text_type=j["text_type"],
        )
        for j in tqdm(jobs, desc="build_prompts")
    ]

    llm_kwargs = dict(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
    )

    llm = LLM(**llm_kwargs)
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=1,
        seed=args.seed,
    )

    outputs = llm.generate(prompts, sp)
    translations = [(o.outputs[0].text or "").strip() for o in outputs]

    out_rows = pack_outputs(
        rows=rows,
        jobs=jobs,
        translations=translations,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        model_path=args.model_path,
        translate_question=args.translate_question,
    )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)
    print(f"[save] translated ({len(out_rows)}) -> {save_path}")

    maybe_export_sft(out_rows, args.sft_save_path, args.pick)
    print("[done]")


if __name__ == "__main__":
    main()

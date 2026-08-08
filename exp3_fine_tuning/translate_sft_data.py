"""
Translate sampled teacher SFT / candidate data into another language.


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

SYSTEM_PROMPT4TRANSLATE = (
    "You are a professional translator. "
    "Translate faithfully; preserve math/LaTeX/code; output only the translation."
)
USER_PROMPT4TRANSLATE = (
    "Translate the following text from {src_lang} to {tgt_lang}.\n"
    "Requirements:\n"
    "- Write the translation in {tgt_lang}.\n"
    "- Keep LaTeX / formulas / \\boxed{...} / code unchanged in structure.\n"
    "- Do not answer the question; only translate.\n"
    "- Output the translation only.\n\n"
    "Text:\n{text}"
)


def load_data(data_path: str | Path) -> list[str],list[str],list[str]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and data, f"empty data: {data_path}"
    # list[dict], dict_keys: question, answer, golden_res
    question_ls = answer_ls = golden_res_ls = []
    for item in data:
        question_ls.append(item['question'])
        anwer_ls.append(item['answer'])
        golden_res_ls.append(item['golden_res'])
    return question_ls, answer_ls, golden_res_ls


def format_prompts(
    text_ls: list[str],
    tokenizer: AutoTokenizer,
    src_lang: str,
    tgt_lang: str,
) -> list[str]:

    formatted = []
    for text in text_ls:
        user = (
            USER_PROMPT4TRANSLATE.replace("{src_lang}", LANG_NAME[src_lang])
            .replace("{tgt_lang}", LANG_NAME[tgt_lang])
            .replace("{text}", text)
        )
        msg = [
            {"role": "system", "content": SYSTEM_PROMPT4TRANSLATE},
            {"role": "user", "content": user},
        ]
        formatted.append(
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        )
    return formatted



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Translate teacher SFT data with vLLM")
    p.add_argument("--model_path", type=str, default=DEFAULT_TRANSLATOR)
    p.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="EN SFT json (question + golden_res + answer)",
    )
    p.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="translated json path; default: teacher/math_{tgt_lang}_n2_sft_translated.json",
    )
    p.add_argument("--src_lang", type=str, default="en", choices=list(LANG_NAME))
    p.add_argument("--tgt_lang", type=str, required=True, choices=[c for c in LANG_NAME if c != "en"])
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument(
        "--max_model_len",
        type=int,
        default=16384,
        help="vLLM context cap; keep modest to fit KV cache",
    )
    p.add_argument("--dtype", type=str, default="bfloat16")
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import set_seed as hf_set_seed
    hf_set_seed(args.seed)

    question_ls, answer_ls, golden_res_ls = load_data(args.data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted_prompts = format_prompts(
        text_ls=golden_res_ls,
        tokenizer=tokenizer,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )

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

    outputs = llm.generate(formatted_prompts, sp)
    translations = [o.outputs[0].text.strip() for o in outputs]
    print('translate done.')
    
    if not args.save_path:
        args.save_path = str(DEFAULT_SAVE_DIR / f"math_{args.tgt_lang}_n2_sft_translated.json")
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print('save done.')


if __name__ == "__main__":
    main()

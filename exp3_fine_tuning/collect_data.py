"""
Offline teacher sampling with local vLLM for later best-of-n + SFT distillation.

Default teacher: DeepSeek-R1-Distill-Qwen-32B

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

DEFAULT_TEACHER_MODEL = "/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-32B"


def load_data(data_path: str | Path, limit: int | None) -> list[dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    assert isinstance(raw_data, list) and raw_data, f"empty data: {data_path}"

    data: list[dict] = []
    for i, item in enumerate(raw_data):
        item = {
            "question": item["question"],
            "answer": item['answer'],
        }
        data.append(item)

    if limit is not None:
        data = data[:limit]
    return data


def format_prompts(
    question_ls: list[str],
    tokenizer: AutoTokenizer,
    language_type: str,
) -> list[str]:
    language = Language(language_type)
    formatted: list[str] = []
    for q in question_ls:
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
    return formatted



def sample_res(
    data: list[dict],
    llm: LLM,
    sp: SamplingParams,
    model_path: str,
    language_type:str,
) -> list[dict]:

    question_ls = [item['question'] for item in data]
    answer_ls = [item['answer'] for item in data]

    formatted_prompts = format_prompts(
        question_ls,
        tokenizer,
        language_type,
    )

    o = llm.generate(
        formatted_prompts,
        sp
    )
    # o[i].outputs[j].text
    out = []
    for question, answer, res_group in zip(question_ls,answer_ls, o):
        for res in res_group:
            msg = {
                'question':question,
                'golden_res':res,
                'answer':answer,
            }
            out.append(msg)
    return out



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
    p.add_argument("--model_path", type=str, default=DEFAULT_TEACHER_MODEL)
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
    p.add_argument("--language_type", type=str, default="en", choices=["en", "zh", "es", "vi", "tr"])
    p.add_argument("--n", type=int, default=4, help="samples per question (for later best-of-n)")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=None, help="optional vLLM context cap")
    p.add_argument("--dtype", type=str, default="bfloat16", help="auto|float16|bfloat16")
    p.add_argument("--limit", type=int, default=None, help="only first N questions (debug)")
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import set_seed as hf_set_seed

    hf_set_seed(args.seed)

    data = load_data(args.data_path, args.limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    llm_kwargs = dict(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype=args.dtype,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.n,
        seed=args.seed,
    )

    o = sample_res(
        data=data,
        llm=llm,
        sp=sp,
        model_path=args.model_path,
        language_type=args.language_type,
    )

    print('sample_res done')

    save_json(o, args.save_path)

    print("all done!")


def best_of_n(sft_data_path:str | Path, reward_model_path:str | Path):
    from transformers import AutoModel
    sft_data_dir = Path(sft_data_path).parent
    reward_model = ''
    pass


if __name__ == "__main__":
    main()

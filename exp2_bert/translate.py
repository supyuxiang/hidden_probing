"""
Translate GLUE SST-2 train/test sentences into zh / es / vi / tr with vLLM.

Examples:
  # translate train+test into all 4 languages
  CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp2_bert/translate.py \
      --model_path /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
      --data_dir /root/autodl-tmp/data/text_classification/glue_sst2 \
      --save_dir /root/autodl-tmp/data/text_classification/glue_sst2 \
      --splits train,validation \
      --langs zh,es,vi,tr

  # smoke
  CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp2_bert/translate.py \
      --splits train --langs zh --limit 8
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

LANG_NAME = {
    "zh": "Simplified Chinese",
    "es": "Spanish",
    "vi": "Vietnamese",
    "tr": "Turkish",
}

SYSTEM_PROMPT = "You are a professional translator."

USER_PROMPT = (
    "Please translate the following English movie-review sentence into {target_language}.\n"
    "Preserve the original meaning and sentiment polarity.\n"
    "Provide the translation only, without any additional explanations or comments.\n\n"
    "Here are some examples:\n\n{few_shot}\n\n"
    "Now translate the following text:\n\n{text}\n\n"
    "Your output:\n"
)

FEW_SHOT = {
    "zh": [
        {
            "src": "it 's a charming and often affecting journey .",
            "tgt": "这是一段迷人且常常动人的旅程。",
        },
        {
            "src": "contains no wit , only labored gags",
            "tgt": "毫无机智，只有勉强的笑料。",
        },
        {
            "src": "the film is a hoot , and is that so wrong ?",
            "tgt": "这部电影很好笑，那又怎样呢？",
        },
    ],
    "es": [
        {
            "src": "it 's a charming and often affecting journey .",
            "tgt": "Es un viaje encantador y a menudo conmovedor.",
        },
        {
            "src": "contains no wit , only labored gags",
            "tgt": "No contiene ingenio, solo chistes forzados.",
        },
        {
            "src": "the film is a hoot , and is that so wrong ?",
            "tgt": "La película es divertidísima, ¿y qué tiene de malo?",
        },
    ],
    "vi": [
        {
            "src": "it 's a charming and often affecting journey .",
            "tgt": "Đó là một hành trình duyên dáng và thường rất cảm động.",
        },
        {
            "src": "contains no wit , only labored gags",
            "tgt": "Không có chút hài hước nào, chỉ toàn trò đùa gượng gạo.",
        },
        {
            "src": "the film is a hoot , and is that so wrong ?",
            "tgt": "Bộ phim rất buồn cười, vậy thì có sai không chứ?",
        },
    ],
    "tr": [
        {
            "src": "it 's a charming and often affecting journey .",
            "tgt": "Bu, büyüleyici ve çoğu zaman etkileyici bir yolculuk.",
        },
        {
            "src": "contains no wit , only labored gags",
            "tgt": "Hiç zekâ yok, yalnızca zorlama şakalar var.",
        },
        {
            "src": "the film is a hoot , and is that so wrong ?",
            "tgt": "Film çok komik; peki bunda ne yanlış var?",
        },
    ],
}


def format_few_shot(lang: str) -> str:
    blocks = []
    for ex in FEW_SHOT[lang]:
        blocks.append(f"Input:\n{ex['src']}\nOutput:\n{ex['tgt']}")
    return "\n\n".join(blocks)


def load_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and len(data) > 0
    assert "sentence" in data[0], f"expected 'sentence' field in {path}"
    return data


def save_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_prompts(
    sentences: list[str],
    lang: str,
    tokenizer: AutoTokenizer,
) -> list[str]:
    fs = format_few_shot(lang)
    target = LANG_NAME[lang]
    prompts: list[str] = []
    for text in sentences:
        content = (
            USER_PROMPT.replace("{target_language}", target)
            .replace("{few_shot}", fs)
            .replace("{text}", text)
        )
        msg = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
        )
    return prompts


def translate_split(
    llm: LLM,
    tokenizer: AutoTokenizer,
    sp: SamplingParams,
    data: list[dict],
    lang: str,
) -> list[dict]:
    sentences = [str(item["sentence"]) for item in data]
    prompts = build_prompts(sentences, lang, tokenizer)
    outputs = llm.generate(prompts, sampling_params=sp)

    translated: list[dict] = []
    for item, out in zip(data, outputs):
        text = out.outputs[0].text.strip()
        row = dict(item)
        row["sentence"] = text
        row["sentence_en"] = item["sentence"]  # keep source for auditing
        translated.append(row)
    return translated


def parse_args():
    p = argparse.ArgumentParser(
        description="Translate SST-2 train/test into zh/es/vi/tr with a 14B vLLM model."
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/models/Qwen2.5-14B-Instruct",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default="/root/autodl-tmp/data/text_classification/glue_sst2",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default="/root/autodl-tmp/data/text_classification/glue_sst2",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="train,validation",
        help="comma-separated: train,test,validation",
    )
    p.add_argument(
        "--langs",
        type=str,
        default="zh,es,vi,tr",
        help="comma-separated target languages",
    )
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="only translate first N examples per split (-1 = all)",
    )
    p.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=0,
        help="0 = use all visible GPUs",
    )
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    for lang in langs:
        assert lang in LANG_NAME, f"unsupported lang={lang}; known={list(LANG_NAME)}"

    tp = args.tensor_parallel_size or max(torch.cuda.device_count(), 1)
    print(f"model={args.model_path} tp={tp} splits={splits} langs={langs}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        n=1,
    )

    for split in splits:
        src_path = data_dir / f"{split}.json"
        data = load_json(src_path)
        if args.limit != -1:
            data = data[: args.limit]
        print(f"[split={split}] N={len(data)} from {src_path}")

        for lang in langs:
            out_path = save_dir / f"{split}_{lang}.json"
            print(f"  -> translating to {lang}, save {out_path}")
            translated = translate_split(llm, tokenizer, sp, data, lang)
            save_json(out_path, translated)
            print(f"  done {lang}: example sentence={translated[0]['sentence'][:80]!r}")

    print("ALL_DONE")


if __name__ == "__main__":
    main()

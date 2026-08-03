"""
Smoke test: translate 10 math questions into Chinese with Qwen2.5-14B-Instruct.

Usage:
    CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/smoke_translate.py \
        --model_path /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
        --data_path /root/hidden_prob/data/math/train.json \
        --n 10
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exp1_math.translate import (
    system_prompt4translate,
    user_prompt4translate,
    format_few_shot,
    few_shot,
    mapping_language,
    extract_trans,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str,
                   default='/root/autodl-tmp/models/Qwen2.5-14B-Instruct')
    p.add_argument('--data_path', type=str,
                   default='/root/hidden_prob/data/math/train.json')
    p.add_argument('--target_language', type=str, choices=['zh', 'es', 'tr', 'vi'],
                   default='zh')
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--max_tokens', type=int, default=1024)
    p.add_argument('--temperature', type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)[:args.n]
    question_ls = [item['question'] for item in data]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        args.model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.9,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=0.95,
        top_k=50,
        n=1,
    )

    target = mapping_language(args.target_language)
    fs = format_few_shot(few_shot[args.target_language])

    formatted = []
    for q in question_ls:
        content = (
            user_prompt4translate
            .replace('{target_language}', target)
            .replace('{few_shot}', fs)
            .replace('{text}', q)
        )
        msg = [
            {'role': 'system', 'content': system_prompt4translate},
            {'role': 'user', 'content': content},
        ]
        formatted.append(
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        )

    outputs = llm.generate(prompts=formatted, sampling_params=sp)

    print(f'\n===== {args.target_language} translation smoke test ({len(question_ls)} items) =====\n')
    for i, (q, out) in enumerate(zip(question_ls, outputs)):
        raw = out.outputs[0].text
        trans = extract_trans(raw)
        print(f'--- [{i + 1}] ---')
        print(f'[EN] {q}')
        print(f'[{args.target_language.upper()}] {trans}')
        print()


if __name__ == '__main__':
    main()

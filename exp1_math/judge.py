'''

CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/judge.py \
    --judge_model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_en_n64_t1.5_tokens1024.json \
    --save_path /root/autodl-tmp/exp1_math/judge/reward_math_en_n64_t1.5_tokens1024.pt

'''


import argparse
import json
import sys
from pathlib import Path
import re
import torch

from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


###############  predefined judge prompts  ###############

system_prompt4judge = (
    'You are a careful math judge. '
    'Extract the solution\'s final answer first; if missing, judge Incorrect. '
    'Otherwise judge by mathematical equivalence to the correct answer.'
)
user_prompt4judge = (
    'I have a full chain-of-thought solution to a math problem that needs verification.\n\n'
    'Correct answer: {answer}\n\n'
    'Solution:\n{res}\n\n'
    'Follow these steps strictly:\n'
    '1) Extract the final generated answer claimed by the Solution '
    '(e.g. from \\boxed{...}, "Final Answer", or the last explicit answer statement). '
    'If no clear final answer can be found, immediately judge Incorrect.\n'
    '2) Only if a generated answer was found, check whether it is mathematically equivalent '
    'to the Correct answer. Allow equivalent transformations '
    '(e.g. 1/2 = 0.5, 2/4 = 1/2, x=3 vs 3, simplified radicals/fractions, reordered terms). '
    'If they are equivalent, mark Correct; otherwise mark Incorrect.\n\n'
    'You may briefly reason step by step first (keep it short and simple). After your reasoning, output the final judgment '
    'on its own last line in exactly one of these two forms:\n'
    'Verdict: Correct\n'
    'or\n'
    'Verdict: Incorrect\n'
    'Do not put any other text after the Verdict line.'
)

########################################################

# Keep the tail: final answer / \\boxed{} is usually at the end.
MAX_RES_TOKENS = 2048


def _truncate_res(res: str, tokenizer: AutoTokenizer, max_tokens: int = MAX_RES_TOKENS) -> str:
    ids = tokenizer.encode(res, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return res
    return tokenizer.decode(ids[-max_tokens:], skip_special_tokens=True)


def _parse_verdict(text: str) -> bool:
    """Parse judge output. Prefer explicit `Verdict:` line; else last Correct/Incorrect."""
    m = re.search(r'(?im)^\s*Verdict\s*:\s*(Correct|Incorrect)\s*$', text)
    if m:
        return m.group(1).lower() == 'correct'
    # fallback: last standalone Correct/Incorrect token (Incorrect first in alternation)
    matches = re.findall(r'(?i)\b(Incorrect|Correct)\b', text)
    if not matches:
        raise ValueError(f'Strange Verify Result: {text!r}')
    return matches[-1].lower() == 'correct'


def judge_math(
    res_ls: list[str],
    answer_ls: list[str],
    llm: LLM,
    tokenizer: AutoTokenizer,
    max_res_tokens: int = MAX_RES_TOKENS,
) -> tuple[list[bool], torch.Tensor]:
    assert hasattr(tokenizer, 'apply_chat_template')
    formatted = []
    for res, answer in zip(res_ls, answer_ls):
        res = _truncate_res(str(res), tokenizer, max_res_tokens)
        message = [
            {"role": "system", "content": system_prompt4judge},
            {"role": "user",'content': user_prompt4judge.replace('{answer}',answer).replace('{res}',res)},
        ]
        formatted.append(tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True))

    sp = SamplingParams(
        max_tokens=4096,
        temperature=0.0,
        top_k=50,
        top_p=0.95,
    )
    outputs = llm.generate(prompts=formatted,sampling_params=sp)
    # outputs[i].outputs[j].text: str
    judge_results = []
    for group in outputs:
        # group_size = 1
        text = group.outputs[0].text.strip()
        judge_results.append(_parse_verdict(text))

    rewards = torch.Tensor(
        [int(b) for b in judge_results]
    ).view(-1,1) # total, 1

    return judge_results, rewards


def judge_math_api(
    res_ls:list[str],
    answer_ls:list[str],
    base_url:str,
    api_key:str,
    model:str='gpt-4o-mini',
    max_tokens:int=512,
    temperature:float=0.0,
    max_workers:int=8,
    max_res_chars: int = 8000,
) -> tuple[list[bool], torch.Tensor]:
    from openai import OpenAI
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    def _judge_one(res:str, answer:str) -> bool:
        res = str(res)
        if len(res) > max_res_chars:
            res = res[-max_res_chars:]
        messages = [
            {"role":"system","content":system_prompt4judge},
            {"role":"user","content":user_prompt4judge.replace('{answer}',answer).replace('{res}',res)},
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = (resp.choices[0].message.content or '').strip()
        return _parse_verdict(text)

    n = len(res_ls)
    judge_results:list[bool] = [False] * n
    pairs = list(zip(res_ls, answer_ls))

    if max_workers and max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {
                ex.submit(_judge_one, res, ans): i for i, (res, ans) in enumerate(pairs)
            }
            for fut in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc='judging (api)'):
                judge_results[future_to_idx[fut]] = fut.result()
    else:
        for i, (res, ans) in enumerate(tqdm(pairs, desc='judging (api)')):
            judge_results[i] = _judge_one(res, ans)

    rewards = torch.Tensor(
        [int(b) for b in judge_results]
    ).view(-1, 1)  # total, 1

    return judge_results, rewards
    
    


def load_data(data_path: str | Path) -> tuple[list[str],list[str],list[str]]:
    question_ls = []
    res_ls = []
    answer_ls = []
    with open(data_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    for i, item in enumerate(data):
        question_ls.append(item['question'])
        answer_ls.append(item['answer'])
        # sampled: res; SFT: golden_res; candidates: res_ls (judge first)
        if "res" in item:
            res_ls.append(item["res"])
        elif "golden_res" in item:
            res_ls.append(item["golden_res"])
        elif "res_ls" in item and item["res_ls"]:
            res_ls.append(item["res_ls"][0])
        else:
            raise KeyError(f"row {i} needs res / golden_res / res_ls; keys={list(item.keys())}")
    return question_ls, answer_ls, res_ls

def save_rewards(rewards:torch.Tensor,save_path:str|Path):
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True,parents=True)
    rewards = rewards.detach().cpu()
    torch.save(rewards,save_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--judge_model", type=str, default="/root/autodl-tmp/models/Qwen2.5-14B-Instruct")
    p.add_argument("--save_path", type=str, default="/root/autodl-tmp/exp1_math/judge/reward_math_test_en_n8_t1.5_tokens1024.pt")
    p.add_argument("--data_path", type=str, default='/root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_en_n8_t1.5_tokens1024.json')
    p.add_argument('--gpu_memory_utilization',type=float,default=0.9)
    return p.parse_args()


def main():
    args = parse_args()
    llm = LLM(model=args.judge_model, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=20000)
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    question_ls,answer_ls,res_ls=load_data(args.data_path)

    judge_resules,rewards = judge_math(
        res_ls=res_ls,
        answer_ls=answer_ls,
        llm=llm,
        tokenizer=tokenizer,
    )
    print('judge done.')

    save_rewards(rewards,args.save_path)
    print('save done.')


if __name__ == "__main__":
    main()

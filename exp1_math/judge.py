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

from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


###############  predefined judge prompts  ###############

system_prompt4judge = 'You are a math expert.'
user_prompt4judge = (
    'I have a full chain-of-thought solution to a math problem that needs verification.'
    'Please read the entire reasoning process and final conclusion, then decide whether the'
    'solution is correct. Ignore minor formatting issues or missing units if the math is correct.'
    'Correct answer: {answer}\n\n'
    'Solution:{res}\n\n'
    'Just answer in one word: "Correct" or "Incorrect", no other words.'
)

########################################################


def judge_math(
    res_ls: list[str],
    answer_ls: list[str],
    llm: LLM,
    tokenizer: AutoTokenizer,
) -> tuple[list[bool], torch.Tensor]:
    assert hasattr(tokenizer, 'apply_chat_template')
    formatted = []
    for res, answer in zip(res_ls, answer_ls):
        message = [
            {"role": "system", "content": system_prompt4judge},
            {"role": "user",'content': user_prompt4judge.replace('{answer}',answer).replace('{res}',res)},
        ]
        formatted.append(tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True))

    sp = SamplingParams(
        max_tokens=20,
        temperature=0.0,
        top_k=50,
        top_p=0.95,
    )
    outputs = llm.generate(prompts=formatted,sampling_params=sp)
    # outputs[i].outputs[j].text: str
    judge_results = []
    for group in outputs:
        # group_size = 1
        text = group.outputs[0].text.lower().strip()
        matches = re.findall(r'incorrect|correct',text)
        if len(matches) != 1:
            final_decision = matches[-1]
        elif len(matches) == 1:
            final_decision = matches[0]
        else:
            raise ValueError(f"Strange Verify Result: {text}")
        judge = (final_decision == 'correct')
        judge_results.append(judge)

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
    max_tokens:int=20,
    temperature:float=0.0,
    max_workers:int=8,
) -> tuple[list[bool], torch.Tensor]:
    from openai import OpenAI
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    def _judge_one(res:str, answer:str) -> bool:
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
        text = (resp.choices[0].message.content or '').lower().strip()
        matches = re.findall(r'incorrect|correct', text)
        if not matches:
            raise ValueError(f"Strange Verify Result: {text!r}")
        # take the last verdict (matches judge_math's behavior on multi-match)
        return matches[-1] == 'correct'

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
    
    


def load_data(data_path: str | Path) -> tuple[list[str],list[str]]:
    question_ls = []
    res_ls = []
    answer_ls = []
    with open(data_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        question_ls.append(item['question'])
        res_ls.append(item['res'])
        answer_ls.append(item['answer'])
    return question_ls, answer_ls, res_ls

def save_rewards(rewards:torch.Tensor,save_path:str|Path):
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True,parents=True)
    rewards = rewards.detach().cpu()
    torch.save(rewards,save_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--judge_model", type=str, default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    p.add_argument("--save_path", type=str, default="/root/autodl-tmp/exp1_math/judge/reward_math.pt")
    p.add_argument("--data_path", type=str, default='/root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math.json')
    return p.parse_args()


def main():
    args = parse_args()
    llm = LLM(model=args.judge_model, tensor_parallel_size=1, gpu_memory_utilization=0.9)
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

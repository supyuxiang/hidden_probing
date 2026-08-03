# translate the data to different languages
import torch
import torch.nn as nn
from pathlib import Path
import sys
import json
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import re



system_prompt4tranlate = (
    'You are a professional translator.'
)

user_prompt4translate = (
    'Please translate the following text into {target_language}:\n\n{text}\n\n'
    'Please provide the translation only, without any additional explanations or comments.'
    'Your outputs:\n\n'
)

# target_language = 'Chinese'

# NOTE: 取第一个\n之后的内容（after），然后去掉首位的\n。
def extract_trans(text):
    _,_,after = text.partition('\n')
    return after.strip()


# TODO
def double_check(raw,trans,llm,):
    system_prompt4double_check = 'You are a professional translator and proofreader.'
    user_prompt4double_check = ''
    return 


def set_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, default='/root/autodl-tmp/models/Qwen2.5-32B-Instruct')
    p.add_argument('--data_path',type=str,default='/root/hidden_prob/data/math/train.json')
    p.add_argument('--target_language',type=str,choices=['zh','es','tr','vi'],default='zh')
    p.add_argument('--max_tokens',type=int,default=4096)
    p.add_argument('--top_p',type=float,default=0.95)
    p.add_argument('--top_k',type=int,default=50)
    p.add_argument('--temperature',type=float,default=0.0)
    p.add_argument('--n',type=int,default=1)
    p.add_argument('--save_path',type=str,default='/root/hidden_prob/data/math/train_zh.json')
    p.add_argument('--limit',type=int,default=-1)
    return p.parse_args()

def main():
    args = set_args()
    # load data
    with open(args.data_path,'r',encoding='utf-8') as f:
        data = json.load(f)[:args.limit] if args.limit != -1 else json.load(f)
    question_ls = [item['question'] for item in data]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    llm = LLM(
        args.model_path,
        tensor_parallel_size=torch.cuda.device_count()
    )
    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        n=args.n,
    )

    formatted = []
    for q in question_ls:
        msg = [
            {'role':'system','content':system_prompt4translate},
            {'role':'user','content':user_prompt4translate.replace('{target_language}',args.target_language).replace('{text}',q)}
        ]
        formatted.append(
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_token=True)
        )
    
    o = llm.generate(formatted,sampling_params=sp)
    trans = []
    for i in range(len(question_ls)):
        raw_trans = o[i].outputs[0].text
        trans.append(
            extract_trans(raw_trans)
        )
    
    out = [{'question': t, 'answer': item['answer']} for t, item in zip(trans, data)]
    with open(args.save_path,'w',encoding='utf-8') as f:
        json.dump(out,f,ensure_ascii=False,indent=2)
    

if __name__ == '__main__':
    main()

    
    
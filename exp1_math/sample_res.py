'''

CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_res.py \
    --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
    --data_path /root/hidden_prob/data/math/test.json \
    --save_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_en_n64_t1.5_tokens1024.json \
    --language_type en \
    --max_tokens 1024 \
    --temperature 1.5 \
    --n 8

'''

import argparse
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

from language import Language



def load_questions_answers(data_path:str|Path) -> tuple[list[str], list[str]]:
    question_ls=[]
    answer_ls=[]
    with open(data_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        question_ls.append(item['question'])
        answer_ls.append(item['answer'])
    
    return question_ls, answer_ls


def format_prompt4generation(question_ls:list[str],tokenizer:AutoTokenizer,language:Language) -> list[str]:
    system_prompt = language.system_prompt
    user_prompt = language.user_prompt
    formatted = []
    for q in question_ls:
        msg = [
            {'role':'system','content':system_prompt},
            {'role':'user','content':user_prompt.replace('{question}',q)}
        ]
        formatted.append(
            tokenizer.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
        )
    return formatted




def sample_res(
    question_ls: list[str],
    answer_ls:list[str],
    tokenizer,
    llm: LLM,
    sp: SamplingParams,
    language_type:str,
) -> list[dict]:
    language = Language(language_type)
    prompt_ls = format_prompt4generation(question_ls, tokenizer,language)
    outputs = llm.generate(prompt_ls, sp)
    # outputs[i].outputs[j].text
    res = [
        [o2.text for o2 in o1.outputs] for o1 in outputs
    ]

    # flatten_res
    flattened = []
    for i,group in tqdm(enumerate(res)):
        question = question_ls[i]
        answer = answer_ls[i]
        for single in group:
            flattened.append(
                {'question':question,'answer':answer,'res':single}
            )
    return flattened


def save_res(flattened: list[list[str]], save_path: str | Path):
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True,parents=True)
    with open(save_path,'w',encoding='utf-8') as f:
        json.dump(flattened,f)
    


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--temperature',type=float,default=1.5)
    p.add_argument('--top_k',type=int,default=50)
    p.add_argument('--top_p',type=float,default=0.95)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument('--data_path',type=str, default='/root/hidden_prob/data/math/train.json') # '/root/hidden_prob/data/math/test.json'
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/models/Qwen2.5-3B-Instruct")
    p.add_argument( "--save_path",type=str,default="/root/autodl-tmp/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math.json")
    p.add_argument('--language_type',type=str,default='en')
    p.add_argument('--n',type=int,default=8,help='number of samples per question')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--gpu_memory_utilization',type=float,default=0.95)
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import set_seed as hf_set_seed
    hf_set_seed(args.seed)
    
    question_ls,answer_ls = load_questions_answers(args.data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n,
    )

    flattened = sample_res(
        question_ls,
        answer_ls,
        tokenizer,
        llm,
        sp,
        args.language_type,
    )
    print('sample and flatten done')

    save_res(flattened,save_path=args.save_path)
    print('save done')
    

if __name__ == "__main__":
    main()

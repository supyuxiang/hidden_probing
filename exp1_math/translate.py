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




def mapping_language(language):
    assert language in ['zh','es','tr','vi']
    return {
        'zh': 'Chinese',
        'es': 'Spanish',
        'tr': 'Turkish',
        'vi': 'Vietnamese',
    }[language]


system_prompt4translate = (
    'You are a professional translator.'
)

user_prompt4translate = (
    'Please translate the following text into {target_language}.\n'
    'Keep all LaTeX mathematical notation exactly as-is (e.g. inline math, fractions, '
    'exponents, subscripts, \\boxed); translate only the natural language.\n'
    'Provide the translation only, without any additional explanations or comments.\n\n'
    'Here are some examples:\n\n{few_shot}\n\n'
    'Now translate the following text:\n\n{text}\n\n'
    'Your output:\n'
)


def format_few_shot(examples: list[dict]) -> str:
    """Render few-shot examples as alternating Input/Output blocks."""
    blocks = []
    for ex in examples:
        blocks.append(f"Input:\n{ex['question']}\nOutput:\n{ex['translation']}")
    return '\n\n'.join(blocks)

few_shot = {
    'zh': [
        {'question': 'How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?',
         'translation': '函数 $y=\\frac{2}{x^2+x-6}$ 的图像有多少条垂直渐近线？'},
        {'question': 'If $5x - 3 = 12$, what is the value of $5x + 3$?',
         'translation': '若 $5x - 3 = 12$，则 $5x + 3$ 的值是多少？'},
        {'question': 'What is the remainder when $2^{100}$ is divided by $7$?',
         'translation': '$2^{100}$ 除以 $7$ 的余数是多少？'},
    ],
    'es': [
        {'question': 'How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?',
         'translation': '¿Cuántas asíntotas verticales tiene la gráfica de $y=\\frac{2}{x^2+x-6}$?'},
        {'question': 'If $5x - 3 = 12$, what is the value of $5x + 3$?',
         'translation': 'Si $5x - 3 = 12$, ¿cuál es el valor de $5x + 3$?'},
        {'question': 'What is the remainder when $2^{100}$ is divided by $7$?',
         'translation': '¿Cuál es el resto cuando $2^{100}$ se divide entre $7$?'},
    ],
    'tr': [
        {'question': 'How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?',
         'translation': '$y=\\frac{2}{x^2+x-6}$ fonksiyonunun grafiğinin kaç tane dikey asimptotu vardır?'},
        {'question': 'If $5x - 3 = 12$, what is the value of $5x + 3$?',
         'translation': '$5x - 3 = 12$ ise, $5x + 3$\'ün değeri nedir?'},
        {'question': 'What is the remainder when $2^{100}$ is divided by $7$?',
         'translation': '$2^{100}$ sayısı $7$\'ye bölündüğünde kalan kaçtır?'},
    ],
    'vi': [
        {'question': 'How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?',
         'translation': 'Đồ thị của hàm $y=\\frac{2}{x^2+x-6}$ có bao nhiêu tiệm cận đứng?'},
        {'question': 'If $5x - 3 = 12$, what is the value of $5x + 3$?',
         'translation': 'Nếu $5x - 3 = 12$, thì giá trị của $5x + 3$ là bao nhiêu?'},
        {'question': 'What is the remainder when $2^{100}$ is divided by $7$?',
         'translation': 'Số dư khi $2^{100}$ chia cho $7$ là bao nhiêu?'},
    ]
}

# target_language = 'Chinese'

# NOTE: 取第一个\n之后的内容（after），然后去掉首位的\n。
def extract_trans(text):
    _,_,after = text.partition('\n')
    return after.strip()


def double_check(raw, trans, llm, tokenizer, target_language='zh'):
    """
    Proofread and refine a candidate translation against its source.
    """
    
    system_prompt4double_check = 'You are a professional translator and proofreader.'
    user_prompt4double_check = (
        'You are proofreading a translation from English into {target_language}.\n'
        'Keep all LaTeX mathematical notation exactly as-is (e.g. inline math, fractions, '
        'exponents, subscripts, \\boxed); only the natural language should be translated.\n\n'
        'Source (English):\n{raw}\n\n'
        'Candidate translation:\n{trans}\n\n'
        'Check the candidate for: (1) faithfulness to the source, '
        '(2) LaTeX / math preserved exactly as in the source, '
        '(3) fluency in {target_language}.\n'
        'If the candidate is already correct, return it unchanged. '
        'Output the final translation only, without any explanations or comments.\n\n'
        'Your output:\n'
    )
    content = (
        user_prompt4double_check
        .replace('{target_language}', mapping_language(target_language))
        .replace('{raw}', raw)
        .replace('{trans}', trans)
    )
    msg = [
        {'role': 'system', 'content': system_prompt4double_check},
        {'role': 'user', 'content': content},
    ]
    prompt = tokenizer.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True
    )
    sp = SamplingParams(temperature=0.0, max_tokens=4096, top_p=0.95, top_k=50, n=1)
    out = llm.generate([prompt], sampling_params=sp)[0]
    return out.outputs[0].text.strip()


def set_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, default='/root/autodl-tmp/models/Qwen2.5-14B-Instruct')
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
    fs = format_few_shot(few_shot[args.target_language])
    for q in question_ls:
        content = (
            user_prompt4translate
            .replace('{target_language}', mapping_language(args.target_language))
            .replace('{few_shot}', fs)
            .replace('{text}', q)
        )
        msg = [
            {'role':'system','content':system_prompt4translate},
            {'role':'user','content':content}
        ]
        formatted.append(
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
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

    
    
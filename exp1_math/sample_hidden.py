'''

CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
    --batch_size 4 \
    --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
    --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_en_n32_t1.5_tokens1024.json \
    --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_train_en_n32_tokens1024.pt \
    --language_type en \
    --layer_indices all \
    --pooling_mode last 

'''

import argparse
import json
import sys
from pathlib import Path
import gc

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exp1_math.language import Language


device = torch.device('cuda')


def get_layer_indices(total_layers:int,layer_ratio_ls:list[float]):
    return [round(ratio * total_layers) for ratio in layer_ratio_ls]

def layer_rato(total_layers:int,layer_indices:list[int]):
    return [float(idx / total_layers) for idx in layer_indices]


def format_prompt4hidden(
    question_ls: list[str],
    res_ls: list[str],
    tokenizer,
    language:Language,
) -> list[str]:

    prompt_ls = []
    for question, res in zip(question_ls, res_ls):
        message = [
            {"role": "system", "content": language.system_prompt},
            {"role": "user", "content": language.user_prompt.replace('{question}',question)},
            {"role": "assistant", "content": res},
        ]
        prompt_ls.append(
            tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=False
            )
        )
    return prompt_ls



def get_pooling_fn(pooling_mode:str):
    if pooling_mode == 'last':
        def _last(x:torch.Tensor,attention_mask:torch.Tensor) -> torch.Tensor:
            # x: batch_size, seq_len, hidden_dim
            # x: batch_size, seq_len
            assert  len(x.shape) == 3 and len(attention_mask.shape)==2
            last_token_indices = (attention_mask.sum(dim=-1) -1).clamp(min=0) # batch_size,
            return x[
                torch.arange(x.size(0)),
                last_token_indices
            ] # batch_size, hidden_dim
        return _last

    elif pooling_mode == 'mean':
        def _mean(x:torch.Tensor,attention_mask:torch.Tensor) -> torch.Tensor:
            # x: batch_size, seq_len, hidden_dim
            # attention_mask: batch_size, seq_len
            num_token = (attention_mask.sum(dim=-1)).clamp(min=1.0).unsqueeze(1) # batch_size,1
            x_valid = x * attention_mask.unsqueeze(-1) # batch_size, seq_len, hidden_dim
            return x_valid.sum(dim=1) / num_token
        return _mean
    
    # elif pooling_mode == 'cls':
    #     # do not used
    #     def _cls(x, attention_mask):
    #         return x[:, 0, :]  # batch_size, hidden_dim
    #     return _cls
        
    else:
        raise NotImplementedError



def sample_hiddens(
    model:AutoModelForCausalLM,
    tokenizer:AutoTokenizer,
    data_ls:list[dict],
    language_type:str,
    layer_indices:int | list[int] | str,
    batch_size:int=4,
    pooling_mode:str='last',
):
    # load question and res
    question_ls=[]
    res_ls=[]
    for item in data_ls:
        question_ls.append(item['question'])
        res_ls.append(item['res'])

    # format prompt
    language = Language(language_type)
    formatted = format_prompt4hidden(question_ls,res_ls,tokenizer,language)

    pooling_fn = get_pooling_fn(pooling_mode)

    # process layer_indices to list
    if isinstance(layer_indices,str):
        if layer_indices == 'all':
            layer_indices = list(range(model.config.num_hidden_layers))
        elif ',' in layer_indices: # '1,2,3,4,5,6' -> [1,2,3,4,5,6]
            layer_indices = sorted([int(idx) for idx in layer_indices.strip().split(',')])
        elif len(layer_indices.strip()) == 1:
            layer_indices = [int(layer_indices)] # [ idx ]
        else:
            print(f'layer_indices:{layer_indices}')
            print(f'type of layer_indices:{type(layer_indices)}')
            raise ValueError
    elif isinstance(layer_indices,list):
        layer_indices.sort()
    elif isinstance(layer_indices,int):
        layer_indices = [layer_indices]
    else:
        raise ValueError


    @torch.inference_mode()
    def process_batch(batch:list[str]):
        # batch: list[str]
        encoded = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
        )
        inputs = {k:v.to(device) for k,v in encoded.items()}
        attention_mask = inputs['attention_mask']
        o = model(**inputs,output_hidden_states=True)
        batch_hs = {
            layer_idx:pooling_fn(o.hidden_states[layer_idx],attention_mask).detach().cpu()
            for layer_idx in layer_indices
        }
        del inputs, o, attention_mask
        torch.cuda.empty_cache()
        gc.collect()
        return batch_hs
    

    total_hs = {
        layer_idx:[]
        for layer_idx in layer_indices
    }

    sample_hs_bar = tqdm(range(0,len(data_ls),batch_size),desc='sampling hs ...')
    for i in sample_hs_bar:
        batch = formatted[i:i+batch_size]
        batch_hs = process_batch(batch)
        for layer_idx in layer_indices:
            total_hs[layer_idx].append(batch_hs[layer_idx])
        del batch_hs, batch
        torch.cuda.empty_cache()
        gc.collect()

    total_hs = {
        layer_idx:torch.cat(total_hs[layer_idx],dim=0) # N, hidden_dim
        for layer_idx in layer_indices
    }
    return total_hs

        
def save_hs(total_hs:dict[int,torch.Tensor],save_path:str | Path):
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True,parents=True)
    torch.save(total_hs,save_path)

def load_data(data_path:str | Path):
    data_path = Path(data_path)
    data_path.parent.mkdir(exist_ok=True,parents=True)
    with open(data_path,'r',encoding='utf-8') as f:
        data_ls = json.load(f)
    return data_ls


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/models/Qwen2.5-3B-Instruct")
    p.add_argument("--save_path", type=str, default="/root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math.pt")
    p.add_argument("--data_path", type=str, default='/root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math.json')
    p.add_argument('--language_type',type=str,default='en')
    p.add_argument('--layer_indices',type=str,default='all')
    p.add_argument('--pooling_mode',type=str,default='last')
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    
    total_hs = sample_hiddens(
        model=model,
        tokenizer=tokenizer,
        data_ls=load_data(args.data_path),
        language_type=args.language_type,
        layer_indices=args.layer_indices,
        batch_size=args.batch_size,
        pooling_mode=args.pooling_mode,
    )
    print('sample done.')

    save_hs(total_hs,args.save_path)
    print('save done.')


if __name__ == "__main__":
    main()

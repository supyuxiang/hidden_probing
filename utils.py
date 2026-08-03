import torch
import torch.nn as nn

from pathlib import Path
import json
import sys

from transformers import AutoTokenizer



def set_seed(seed:int=42):
    import numpy as np
    import torch
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_tokenizer(model_path:str|Path) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_path)
    assert hasattr(tok,'apply_chat_template')
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_json(path: str | Path) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_jsonl(path: str | Path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [
            json.loads(line.strip()) for line in f if line.strip()
        ]
    return data

def save_model_state(model, path: str | Path):
    torch.save(model.state_dict(),path)

def reload_model_state(model, path: str | Path):
    model.load_state_dict(torch.load(path))

    


















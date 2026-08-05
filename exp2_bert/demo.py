import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertTokenizer, BertModel

from pathlib import Path
from tqdm import tqdm
import json
import sys


model_path = Path('/root/autodl-tmp/bert-base-multilingual-cased')
device = torch.device('cuda')

sentence = [
    'Hello! How are you?',
    'I am fine, thank you!'
]

# load_model

tokenizer = BertTokenizer.from_pretrained(model_path)

model = BertModel.from_pretrained(model_path)
model.to(device)

inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
inputs = {k: v.to(device) for k, v in inputs.items()}

outputs = model(**inputs, output_hidden_states=True)

hs = outputs.hidden_states
breakpoint()
print(hs)










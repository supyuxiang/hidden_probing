"""BERT sentence encoding demo: use [CLS] as the sentence representation.

Classic HuggingFace path (BertModel):
  - outputs.last_hidden_state[:, 0, :]  == [CLS]
  - outputs.pooler_output is CLS after Linear+Tanh (NSP head); often worse as a
    general sentence embedding than the raw CLS hidden state.

Better sentence-embedding API (optional):
  pip install sentence-transformers
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
  emb = model.encode(sentences)  # typically mean-pooling + trained for similarity
"""

from pathlib import Path

import torch
from transformers import BertModel, BertTokenizer

model_path = Path("/root/autodl-tmp/models/bert-base-multilingual-cased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentences = [
    "Hello! How are you?",
    "I am fine, thank you!",
]

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)
model.to(device)
model.eval()

inputs = tokenizer(
    sentences,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# [CLS] = first token of the last layer (standard BERT sentence vector for classification)
cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
# Equivalent: outputs.hidden_states[-1][:, 0, :]

# Official pooler: CLS -> Linear -> Tanh (pretrained for NSP, not always best)
pooler_emb = outputs.pooler_output  # (batch, hidden)

print("device:", device)
print("cls_emb shape:", tuple(cls_emb.shape))
print("pooler_emb shape:", tuple(pooler_emb.shape))
print("cls_emb[0, :8]:", cls_emb[0, :8])

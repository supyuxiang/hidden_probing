import torch
import torch.nn as nn
from pathlib import Path
import sys
from torch.utils.data import Dataset, DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type=='cpu': print('using cpu')


def load_hs(hs_path:str|Path):
    obj = torch.load(hs_path, weight_only=True, mmap=True)
    assert isinstance(obj, dict)
    hs = {int(k):v for k,v in obj.items()} # layer_idx: (N, hidden_dim)
    return hs








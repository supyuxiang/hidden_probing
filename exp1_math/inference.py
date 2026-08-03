import torch
import torch.nn as nn
from pathlib import Path
import sys
from torch.utils.data import Dataset, DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type=='cpu': print('using cpu')


def load_hs(hs_path:str|Path):
    hs = torch.load(hs_path,weight_only=True)
    assert len(hs.shape) == 2, f'shape of hs: {hs.shape}, expected 2D tensor'
    # hs: batch_size, hidden_dim
    return hs








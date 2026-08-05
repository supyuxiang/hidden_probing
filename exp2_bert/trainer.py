import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from pathlib import Path
import sys
from omegaconf import OmegaConf
from tqdm import tqdm
import hydra
from transformers import BertTokenizer, BertModel, AutoTokenizer




class BertDataset(Dataset):
    def __init__(
        self,
        
    ):
        super(BertDataset, self).__init__()
        pass
    
    def __len__(self):
        return
    
    def __getitem__(self,idx:int):
        return 
        

class Trainer:
    def __init__(
        self,
        config,
    ):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu': print('using cpu!')
        self.generator = torch.Generator().manual_seed(self.config.seed)

        self.build_dataloader()
        self.build_model()
        self.build_optimizer()
        self.build_scheduler()
        self.build_loss_fn()

    def build_dataloader(self):
        pass
    
    def build_model(self):
        pass
    
    def build_optimizer(self):
        pass
    
    def build_scheduler(self):
        pass
    
    def build_loss_fn(self):
        pass
    
    def train(self):
        pass
    
    def train_epoch(self):
        pass

    def eval_epoch(self):
        pass
    
    def plt_fig(self):
        pass


@hydra.main(config_path='.',config_name='config')
def main(cfg):
    config = omegaconf.create(cfg)
    pass

if __name__ == "__main__":
    main()








import json
import random
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer
from omegaconf import OmegaConf
import numpy as np
import sys
import math
from torch.utils.data import Dataset, DataLoader
import hydra
import argparse

ROOT = Path(__file__).parent.parent
sys.path.insert(0,str(ROOT))



class SFTDataset(Dataset):
    def __init__(
        self,
        config,
    ):
        super(SFTDataset, self).__init__()
        
    def __len__(self):
        return 
    
    def __getitem__(self,idx:int):
        return
    
def colalte_fn(batch):
    return




class Trainer:
    def __init__(
        self,
        config,
    ):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu': print('using cpu!')
        self.base_generator = torch.Generator().manual_seed(self.config.seed)

        self._init_swanlab()
        self.build_dataloader()
        self.load_model()
        self.build_optimizer()
        self.build_scheduler()
        self.build_loss_fn()
        
    
    def _init_swanlab(self):
        self.run = None
        pass

    
    def build_dataloader(self):
        langs = self.config.languages.strip().lower().split(',')
        num_lang = len(langs)
        self.sub_lang_size = round(self.config.total_size / num_lang)
        train_data = []
        self.dataset_val: dict = {}
        for i,lang in enumerate(langs):
            gen = torch.Generator().manual_seed(42 + i)
            data_path = Path(self.config.dataset.train_data_dir) / f''
            with open(data_path, 'r', encoding='utf-8') as f:
                obj_train = json.load(f)
            perm = torch.randperm(len(obj_train), generator=gen)[: self.sub_lang_size].tolist()
            train_data.extend(obj_train[i] for i in perm)

            # load val dataset for eval
            val_data_path = Path(self.config.dataset.val_data_dir) / f''
            with open(val_data_path.'r',encoding='utf-8') as f1:
                obj_val = json.load(f1)
            self.dataset_val[lang] = build_dataset(obj_val)
        self.dataset_train = build_dataset(train_data)
        print(f'train_data_size:{len(self.dataset_train)}\nval_data_size:{sum(len(data) for data in self.dataset_val.values() )}')
    
    def load_model(self):
        model_path = self.config.model.model_path
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

        if self.config.train.use_lora:
            lora_cfg = LoraConfig(self.config.train.lora_config)
            self.model.apply_lora(lora_cfg)
    
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
    
    def val_epoch(self):
        pass
    
    def plt_fig(self):
        pass
    
    





def set_args():
    p = argparse.ArgumentPaser()
    return p.parse_args()


def mian():
    pass












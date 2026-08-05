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
from transformers import BertTokenizer, BertModel



class Classifier_Bert(nn.Module):
    def __init__(
        self,
        bert_path:str|Path,
        input_dim:int,
        output_dim:int,
        layer_idx:int,
        pooling_mode:str='last',
        device:torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super(Classifer_Bert,self).__init__()
        self.bert_path = bert_path
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_idx = layer_idx
        self.pooling_mode = pooling_mode
        self.device = device

        self._load_bert()
        self.classifier_head = nn.Linear(self.input_dim,self.output_dim)
        nn.init.xavier_normal_(self.classifier_head.weight)
        if hasattr(self.classifier_head, 'bias') and self.classifier_head.bias is not None:
            nn.init.zeros_(self.classifier_head.bias)

    def _load_bert(self):
        from transformers import BertModel, BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert = BertModel.from_pretrained(self.bert_path)
        self.bert.to(self.device)
    
    def forward(self, texts: str | list[str]):
        if isinstance(texts,str):
            texts = [texts]
        assert isinstance(texts,list)

        encoded = self.tokenizer(texts, return_tensors=True, padding=True, truncation=True)
        inputs = {k:v.to(self.device) for k,v in encoded.items()}
        attention_mask = inputs['attention_mask']

        o = self.model(**inputs, output_hidden_states=True)
        hs = o.hidden_states[layer_idx] # batch_size, seq_len, hidden_dim

        pooled = self.get_pooling_fn()(hs,attention_mask) # batch_size, hidden_dim
        logits = self.classifier_head(pooled) # batch_size, output_dim
        # prob = F.softmax(logits,dim=-1)
        return logtis

    def get_pooling_fn(self):
        if self.pooling_mode == 'last':
            def _last(hs:torch.Tensor,attention_mask:torch.Tensor) -> torch.Tensor:
                return 
            return _last
        elif self.pooling_mode == 'mean':
            def _mean(hs:torch.Tensor, attention_mask:torch.Tensor) -> torch.Tensor:
                return 
            return _mean
        else:
            raise NotImplementedError


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
        self.tokenizer = BertTokenizer.from_pretrained(self.config.model.model_path)
        self.model = BertModel.from_pretrained(self.config.model.model_path)
        self.model.to(self.device)
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]

    def build_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
        )
    
    def build_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.scheduler.T_max,
            eta_min=self.config.scheduler.eta_min,
        )
    
    def build_loss_fn(self):
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
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








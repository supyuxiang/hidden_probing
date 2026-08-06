import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

sys.path.insert(0, str(ROOT))

from exp2_bert.trainer import Trainer



def format_config(lang:str):
    assert lang in ['en','zh','es','vi','tr']
    config_template = {
        'seed':42,
        'exp_name': 'fine_tuning_bert4classification'
    }



def main():
    for lang in langs:
        config = 
        trainer = Trainer(config)
        trainer.train()
        

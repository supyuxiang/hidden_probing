import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

sys.path.insert(0, str(ROOT))

from exp2_bert.trainer import Trainer
from omegaconf import OmegaConf


def format_config(lang: str):
    assert lang in ['en', 'zh', 'es', 'vi', 'tr']
    base_cfg_path = Path(__file__).parent.parent / "config.yaml"
    base = OmegaConf.load(str(base_cfg_path))
    cfg = OmegaConf.create(base)

    if lang == 'en':
        return cfg
    
    cfg.dataset.train_path = str(Path(cfg.dataset.train_path).parent / f"train_{lang}.json")
    cfg.dataset.test_path = str(Path(cfg.dataset.test_path).parent / f"validation_{lang}.json")

    # update experiment name and save dir per language
    cfg.exp_name = f"{cfg.exp_name}_{lang}"
    if 'train' in cfg and 'save_dir' in cfg.train:
        cfg.train.save_dir = str(Path(cfg.train.save_dir).parent / f'bert_tuned_with_{lang}')

    return cfg


def main():
    langs = ['en', 'zh', 'es', 'vi', 'tr']
    for lang in langs:
        print(f"=== Training for language: {lang} ===")
        config = format_config(lang)
        trainer = Trainer(config)
        trainer.train()


if __name__ == '__main__':
    main()
        

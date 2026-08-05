# NOTE: for capability probing only. Now, don't be used.

import argparse
import json
import sys
from pathlib import Path
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from omegaconf import OmegaConf
import swanlab
import matplotlib.pyplot as plt
import hydra


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exp1_math.dataset import ProbeDataset, collate_fn
from exp1_math.model import Classifier_Linear


def set_seed(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(
        self,
        config,
    ):
        self.config = config
        # set_seed(self.config.seed)
        self.generator = torch.Generator().manual_seed(self.config.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu': print('Using CPU')
        
        self._init_swanlab()
        self.build_dataloader()
        self.build_model()
        self.build_optimizer()
        self.build_scheduler()
        self.build_loss_fn()
    
    def _init_swanlab(self):
        config_dict = OmegaConf.to_container(self.config, resolve=True, enum_to_str=True)
        self.run = swanlab.init(
            project='ICLR27',
            experiment_name=self.config.get('exp_name', 'linear_probing'),
            config=config_dict,
        )
    
    def build_dataloader(self):
        self.dataset_train = ProbeDataset(
            self.config.dataset.hiddens_path_train,
            self.config.dataset.reward_path_train,
            self.config.dataset.layer_idx
        )
        self.dataset_test = ProbeDataste(
            self.config.dataset.hiddens_path_test,
            self.config.dataset.reward_path_test,
            self.config.dataset.layer_idx
        )

        self.dataloader_train = DataLoader(
            self.dataset_train,
            self.config.train.batch_size,
            shuffle=True,
            generator=self.generator,
            collate_fn=collate_fn,
        )
        self.dataloader_test = DataLoader(
            self.dataset_test,
            self.config.train.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    
    def build_model(self):
        self.model = Classifier_Linear(
            input_dim=self.config.model.input_dim,
            output_dim=self.config.model.output_dim,
        )
        self.model.to(self.device)

        self.total_params=[p for p in self.model.parameters()]
        self.trainable_params=[p for p in self.model.parameters() if p.requires_grad]
        self.frozen_params=[p for p in self.model.parameters() if not p.requires_grad]
    
    def build_optimizer(self):
        # default : adamw
        if self.config.optimizer.optimizer_type=='AdamW':
            self.optimizer = torch.optim.AdamW(
                self.trainable_params,
                lr=self.config.optimizer.AdamW.lr,
                weight_decay=self.config.optimizer.AdamW.weight_decay,
            )
        elif self.config.optimizer.optimizer_type=='SGD':
            self.optimizer = torch.optim.SGD(
                self.trainable_params,
                lr=self.config.optimizer.SGD.lr,
                weight_decay=self.config.optimizer.SGD.weight_decay
            )
        else:
            raise NotImplementedError
    
    def build_scheduler(self):
        # default : cosine
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.scheduler.T_max,
            eta_min=self.config.scheduler.eta_min
        )
    
    def build_loss_fn(self):
        train_rewards = self.dataset.rewards[self.dataset_train.indices].float()
        n_pos = (train_rewards == 1).sum().clamp(min=1)
        n_neg = (train_rewards == 0).sum().clamp(min=1)
        pos_weight = torch.tensor([n_neg / n_pos], device=self.device)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def train(self):
        num_epochs = self.config.train.num_epochs
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float('inf')
        self.best_state_path = save_dir / 'best_model.pt'

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            test_loss, test_mae, test_acc = self.eval_epoch(epoch)

            self.scheduler.step()

            self.run.log({
                'train/loss': train_loss,
                'val/loss': val_loss,
                'val/mae': val_mae,
                'val/acc': val_acc,
            }, step=epoch)

            print(
                f'Epoch {epoch}/{num_epochs} | '
                f'train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | '
                f'val_mae={val_mae:.4f} | val_acc={val_acc:.4f}'
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_state_path)

        # reload
        self.model.load_state_dict(torch.load(self.best_state_path, map_location=self.device))
        self.plt_fig()
        self.run.finish()

    def train_epoch(self, epoch: int = 0) -> float:
        self.model.train()
        total_loss, n_samples = 0.0, 0
        train_bar = tqdm(
            self.dataloader_train,
            desc=f'Epoch {epoch} [train]',
            leave=False,
        )
        for batch_hs, batch_rewards in train_bar:
            batch_hs = batch_hs.to(self.device)
            batch_rewards = batch_rewards.to(self.device).float()

            logits = self.model(batch_hs)
            loss = self.loss_fn(logits, batch_rewards)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            bs = batch_rewards.size(0)
            total_loss += loss.item() * bs
            n_samples += bs
            train_bar.set_postfix(loss=f'{total_loss / n_samples:.4f}')

        return total_loss / max(n_samples, 1)

    @torch.no_grad()
    def eval_epoch(self, epoch: int = 0) -> tuple[float, float, float]:
        self.model.eval()
        total_loss, total_abs, n_samples = 0.0, 0.0, 0
        correct = 0
        eval_bar = tqdm(
            self.dataloader_test,
            desc=f'Epoch {epoch} [val]',
            leave=False,
        )
        for batch_hs, batch_rewards in eval_bar:
            batch_hs = batch_hs.to(self.device)
            batch_rewards = batch_rewards.to(self.device).float()

            logits = self.model(batch_hs)
            loss = self.loss_fn(logits, batch_rewards)
            probs = torch.sigmoid(logits)
            total_loss += loss.item() * batch_rewards.size(0)
            total_abs += (probs - batch_rewards).abs().sum().item()
            # rewards are 0/1 -> threshold logit at 0 (== prob >= 0.5)
            correct += ((logits >= 0).float() == batch_rewards).sum().item()
            n_samples += batch_rewards.size(0)
            eval_bar.set_postfix(loss=f'{total_loss / n_samples:.4f}')

        mse = total_loss / max(n_samples, 1)
        mae = total_abs / max(n_samples, 1)
        acc = correct / max(n_samples, 1)
        return mse, mae, acc

    @torch.no_grad()
    def plt_fig(self):
        self.model.eval()
        all_preds, all_targets = [], []
        for batch_hs, batch_rewards in self.dataloader_test:
            batch_hs = batch_hs.to(self.device)
            batch_rewards = batch_rewards.to(self.device)
            logits = self.model(batch_hs)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
            all_targets.append(batch_rewards.cpu())

        preds = torch.cat(all_preds).view(-1).numpy()
        targets = torch.cat(all_targets).view(-1).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(targets, preds, s=8, alpha=0.4)
        axes[0].plot([0, 1], [0, 1], 'r--', linewidth=1)
        axes[0].set_xlabel('true reward')
        axes[0].set_ylabel('predicted prob')
        axes[0].set_title('pred vs true')
        axes[1].hist(preds - targets, bins=50)
        axes[1].set_xlabel('error (prob - true)')
        axes[1].set_title('error distribution')
        fig.tight_layout()

        save_dir = Path(self.config.save_dir)
        fig_path = save_dir / 'pred_vs_true.png'
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        self.run.log({'fig/pred_vs_true': swanlab.Image(str(fig_path))})


@hydra.main(config_name='config',config_path='.',version_base=None)
def main(cfg):
    config = OmegaConf.create(cfg)
    trainer = Trainer(config)
    trainer.train()



if __name__ == '__main__':
    main()

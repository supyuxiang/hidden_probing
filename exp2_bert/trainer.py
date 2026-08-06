"""
  source /root/autodl-tmp/miniconda3/bin/activate bert1
  cd /root/hidden_prob/exp2_bert
  python trainer.py
  
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_pooling_fn(pooling_mode: str):
    if pooling_mode == "cls":
        def _cls(hs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            # hs: (B, T, H)
            return hs[:, 0, :]

        return _cls

    if pooling_mode == "mean":
        def _mean(hs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            mask = attention_mask.unsqueeze(-1).to(dtype=hs.dtype)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return (hs * mask).sum(dim=1) / denom

        return _mean

    raise NotImplementedError(f"pooling_mode={pooling_mode}")


class Classifier_Bert(nn.Module):
    """BertModel + linear head on pooled hidden states at a chosen layer."""

    def __init__(
        self,
        bert_path: str | Path,
        output_dim: int,
        layer_idx: int = -1,
        pooling_mode: str = "cls",
        freeze_bert: bool = False,
        max_length: int = 512,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.bert_path = str(bert_path)
        self.output_dim = output_dim
        self.layer_idx = layer_idx
        self.pooling_mode = pooling_mode
        self.max_length = max_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cpu': print('using cpu')

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert = BertModel.from_pretrained(self.bert_path)
        self.hidden_dim = int(self.bert.config.hidden_size)
        self.num_hidden_layers = int(self.bert.config.num_hidden_layers)

        # resolve negative layer index against hidden_states length (emb + layers)
        n_hs = self.num_hidden_layers + 1
        if self.layer_idx < 0:
            self.layer_idx = n_hs + self.layer_idx
        assert 0 <= self.layer_idx < n_hs, (
            f"layer_idx={self.layer_idx} out of range [0, {n_hs})"
        )

        self.pooling_fn = get_pooling_fn(self.pooling_mode)
        self.classifier_head = nn.Linear(self.hidden_dim, self.output_dim)
        nn.init.xavier_normal_(self.classifier_head.weight)
        if self.classifier_head.bias is not None:
            nn.init.zeros_(self.classifier_head.bias)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.to(self.device)

    def forward(self, texts: str | list[str]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        assert isinstance(texts, list)

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in encoded.items()}
        attention_mask = inputs["attention_mask"]

        o = self.bert(**inputs, output_hidden_states=True)
        hs = o.hidden_states[self.layer_idx]  # (B, T, H)
        pooled = self.pooling_fn(hs, attention_mask)  # (B, H)
        logits = self.classifier_head(pooled)  # (B, C)
        return logits


class BertDataset(Dataset):
    """JSON list dataset: each item has a text field and a label field."""

    def __init__(
        self,
        data_path: str | Path,
        text_field: str = "question",
        label_field: str = "subject",
        label2id: dict[str, int] | None = None,
    ):
        super().__init__()
        data_path = Path(data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        assert isinstance(raw, list) and len(raw) > 0

        self.texts: list[str] = []
        labels_raw: list[str] = []
        for item in raw:
            if text_field == "qa":
                text = f"{item['question']} {item['answer']}"
            else:
                text = str(item[text_field])
            self.texts.append(text)
            labels_raw.append(str(item[label_field]))

        if label2id is None:
            classes = sorted(set(labels_raw))
            self.label2id = {c: i for i, c in enumerate(classes)}
        else:
            self.label2id = dict(label2id)
            missing = set(labels_raw) - set(self.label2id)
            assert not missing, f"labels missing from label2id: {missing}"

        self.id2label = {i: c for c, i in self.label2id.items()}
        self.labels = torch.tensor(
            [self.label2id[x] for x in labels_raw], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch: list[tuple[str, torch.Tensor]]):
    texts, labels = zip(*batch)
    return list(texts), torch.stack(list(labels), dim=0)


class Trainer:
    def __init__(self, config):
        self.config = config
        set_seed(int(self.config.seed))
        self.generator = torch.Generator().manual_seed(int(self.config.seed))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            print("using cpu!")

        self.build_dataloader()
        self.build_model()
        self.build_optimizer()
        self.build_scheduler()
        self.build_loss_fn()

        self.history = {"train_loss": [], "train_acc": [], "eval_loss": [], "eval_acc": []}

    def build_dataloader(self):
        ds_cfg = self.config.dataset
        self.dataset_train = BertDataset(
            ds_cfg.train_path,
            text_field=ds_cfg.text_field,
            label_field=ds_cfg.label_field,
        )
        self.dataset_test = BertDataset(
            ds_cfg.test_path,
            text_field=ds_cfg.text_field,
            label_field=ds_cfg.label_field,
            label2id=self.dataset_train.label2id,
        )
        self.num_classes = len(self.dataset_train.label2id)
        print(
            f"[data] train={len(self.dataset_train)} test={len(self.dataset_test)} "
            f"num_classes={self.num_classes} labels={self.dataset_train.label2id}"
        )

        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=int(self.config.train.batch_size),
            shuffle=True,
            generator=self.generator,
            collate_fn=collate_fn,
        )
        self.dataloader_test = DataLoader(
            self.dataset_test,
            batch_size=int(self.config.train.batch_size),
            shuffle=False,
            collate_fn=collate_fn,
        )

    def build_model(self):
        mcfg = self.config.model
        self.model = Classifier_Bert(
            bert_path=mcfg.model_path,
            output_dim=self.num_classes,
            layer_idx=int(mcfg.layer_idx),
            pooling_mode=str(mcfg.pooling_mode),
            freeze_bert=bool(mcfg.freeze_bert),
            max_length=int(mcfg.max_length),
            device=self.device,
        )
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in self.trainable_params)
        n_all = sum(p.numel() for p in self.model.parameters())
        print(
            f"[model] layer_idx={self.model.layer_idx} pooling={self.model.pooling_mode} "
            f"freeze_bert={mcfg.freeze_bert} trainable={n_train}/{n_all}"
        )

    def build_optimizer(self):
        self.optimizer = AdamW(
            self.trainable_params,
            lr=float(self.config.optimizer.lr),
            weight_decay=float(self.config.optimizer.weight_decay),
        )

    def build_scheduler(self):
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=int(self.config.scheduler.T_max),
            eta_min=float(self.config.scheduler.eta_min),
        )

    def build_loss_fn(self):
        self.loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        pred = logits.argmax(dim=-1)
        return (pred == labels).float().mean().item()

    def train_epoch(self) -> tuple[float, float]:
        self.model.train()
        # if bert frozen, still need train mode for dropout in head only;
        # keep bert in eval to disable its dropout when frozen.
        if self.config.model.freeze_bert:
            self.model.bert.eval()

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        bar = tqdm(self.dataloader_train, desc="train", leave=False)
        for texts, labels in bar:
            labels = labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(texts)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()

            acc = self._accuracy(logits.detach(), labels)
            total_loss += loss.item()
            total_acc += acc
            n_batches += 1
            bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

        self.scheduler.step()
        return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)

    @torch.no_grad()
    def eval_epoch(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        bar = tqdm(self.dataloader_test, desc="eval", leave=False)
        for texts, labels in bar:
            labels = labels.to(self.device)
            logits = self.model(texts)
            loss = self.loss_fn(logits, labels)
            pred = logits.argmax(dim=-1)
            total_loss += loss.item() * labels.size(0)
            total_correct += int((pred == labels).sum().item())
            total_n += labels.size(0)
        return total_loss / max(total_n, 1), total_correct / max(total_n, 1)

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "label2id": self.dataset_train.label2id,
                "config": OmegaConf.to_container(self.config, resolve=True),
            },
            path,
        )

    def train(self) -> None:
        best_acc = -1.0
        save_dir = Path(self.config.train.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, int(self.config.train.num_epochs) + 1):
            train_loss, train_acc = self.train_epoch()
            eval_loss, eval_acc = self.eval_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["eval_loss"].append(eval_loss)
            self.history["eval_acc"].append(eval_acc)
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"[epoch {epoch}/{self.config.train.num_epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f} | lr={lr:.2e}"
            )
            if eval_acc > best_acc:
                best_acc = eval_acc
                self.save_checkpoint(save_dir / "best.pt")
                print(f"  -> new best eval_acc={best_acc:.4f}, saved best.pt")

        self.save_checkpoint(save_dir / "last.pt")
        print(f"[done] best_eval_acc={best_acc:.4f} save_dir={save_dir}")


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()

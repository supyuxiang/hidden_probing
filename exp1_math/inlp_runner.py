"""
Iterative Null-space Projection (INLP) for removing language-predictive
directions.

At each iteration i:
  1. Train a linear classifier w_i to predict a (binary) protected attribute
     from hidden states H (one-vs-rest for a given language).
  2. Project H onto the null space of w_i:
        P_t      = I - w_i w_i^T / ||w_i||^2
        H_{i+1}  = H_i @ P_t
    H: (n, d)
    P: (d, d)
    w: (d,)
  3. Accumulate P_perp = P_1 @ ... @ P_t  (cumulative orthogonal projection).
Iterate until the classifier accuracy falls to chance (or T iterations).

Returns:
  H_proj  : (n, d) hidden states after removing all identified language directions.
  P_perp  : (d, d) cumulative projection P_1 @ ... @ P_T (apply to fresh H with H @ P_perp).
  P_lang  : (k, d) stacked removed direction vectors w_i (rowspace = language subspace).
  accs    : list[float] per-iteration classifier accuracy (information removed).
"""

import argparse
import csv
import gc
import sys
from pathlib import Path
import json
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from omegaconf import OmegaConf
from dataclasses import dataclass
import swanlab

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# from exp1_math.trainer import Trainer

LANG2ID = {'en': 0, 'zh': 1, 'es': 2, 'vi': 3, 'tr': 4}
ID2LANG = {v: k for k, v in LANG2ID.items()}


class BinaryLinearClassifier(nn.Module):
    """Single-layer logistic regression: logit = H @ w + b (w is the protected direction)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # weight: (1, d), bias: (1,)
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_normal_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return self.linear(H)  # (n, 1)

    def query_weight(self) -> torch.Tensor:
        """Return the protected direction as a 1-D vector (d,)."""
        return self.linear.weight.detach().squeeze(0).contiguous()  # (d,)

    def count_params(self) -> int:
        return self.linear.weight.numel() + (
            self.linear.bias.numel() if self.linear.bias is not None else 0
        )


class BinaryLinearDataset(Dataset):
    def __init__(self, hs: torch.Tensor, y: torch.Tensor):
        super().__init__()
        self.hs = hs.contiguous().view(hs.shape[0], -1)
        self.y = y.contiguous().view(-1, 1).float()
        assert self.hs.shape[0] == self.y.shape[0]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return self.hs[idx], self.y[idx]


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    batch_hs,batch_y = [],[]
    for hs,y in zip(*batch):
        batch_hs.append(hs)
        batch_y.append(y)
    return torch.stack(batch_hs, dim=0), torch.stack(batch_y, dim=0) # (batch_size, d), (batch_size, 1)


def print_if_verbose(verbose:bool,text:str) -> None:
    if verbose: print(text)


class INLP_Runner:
    def __init__(
        self,
        H: torch.Tensor,
        y: torch.Tensor,
        T: int = 15,
        epochs_per_iter: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        chance_tolerance: float = 0.02,
        batch_size: int = 64,
        verbose: bool = True,
        split_ratio: float = 0.95,
        seed: int = 42,
    ):
        self.H = H.float().contiguous().view(H.shape[0], -1)
        self.y = y.long().contiguous().view(-1)
        assert self.H.shape[0] == self.y.shape[0]

        self.T = T
        self.epochs_per_iter = epochs_per_iter
        self.lr = lr
        self.weight_decay = weight_decay
        self.chance_tolerance = chance_tolerance
        self.batch_size = batch_size
        self.verbose = verbose
        self.split_ratio = split_ratio
        self.seed = seed

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu': print('using cpu')

        self.generator = torch.Generator().manual_seed(self.seed)

        n = len(self.y)
        self.train_size = max(1, round(n * self.split_ratio))
        self.test_size = max(1, n - self.train_size)

        self.H_history: list[torch.Tensor] = []
        self.P_history: list[torch.Tensor] = []

        self._init_swanlab()
        self.build_loss_fn()
        self._rebuild_for_H(self.H)
    
    def _init_swanlab(self):
        pass

    def build_loss_fn(self):
        n_pos = (self.y == 1).sum().clamp(min=1).float()
        n_neg = (self.y == 0).sum().clamp(min=1).float()
        self.pos_weight = (n_neg / n_pos).to(self.device).view(1)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def _rebuild_for_H(self, H: torch.Tensor):
        """Point dataloaders at H and re-init a fresh linear classifier + optim."""
        self.H = H.detach().float().contiguous().view(H.shape[0], -1)
        self.dataset = BinaryLinearDataset(self.H, self.y.float())
        self.dataset_train = Subset(self.dataset, self.train_indices)
        self.dataset_test = Subset(self.dataset, self.test_indices)
        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(self.seed),
        )
        self.dataloader_test = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        self.build_classifier()
        self.build_optimizer()
        self.build_scheduler()

    def build_classifier(self):
        self.classifier = BinaryLinearClassifier(input_dim=self.H.shape[1]).to(self.device)
        self.trainable_params = [p for p in self.classifier.parameters() if p.requires_grad]

    def build_optimizer(self):
        self.optimizer = AdamW(
            self.trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def build_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs_per_iter,
            eta_min=0.0,
            last_epoch=-1,
        )

    def train_language_classifier_epoch(self) -> float:
        self.classifier.train()
        epoch_loss = 0.0
        loader = self.dataloader_train
        it = loader if not self.verbose else tqdm(loader, desc='Training Classifier', leave=False)
        for batch_hs, batch_y in it:
            batch_hs = batch_hs.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            logits = self.classifier(batch_hs)
            loss = self.loss_fn(logits, batch_y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / max(len(loader), 1)

    def eval_language_classifier_epoch(self) -> tuple[float, float]:
        self.classifier.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_hs, batch_y in self.dataloader_test:
                batch_hs = batch_hs.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                logits = self.classifier(batch_hs)
                loss = self.loss_fn(logits, batch_y)
                pred = (torch.sigmoid(logits) >= 0.5).float()
                correct += (pred == batch_y).sum().item()
                total += batch_y.numel()
                total_loss += loss.item() * batch_y.shape[0]
        n = max(total, 1)
        return total_loss / n, correct / n

    # for languages classification
    def train_language_classifier(self, desc: str = 'epochs') -> tuple[float, float, float]:
        train_loss = val_loss = val_acc = 0.0
        epoch_bar = tqdm(
            range(self.epochs_per_iter),
            desc=desc,
            leave=False,
            dynamic_ncols=True,
        )
        for ep in epoch_bar:
            train_loss = self.train_language_classifier_epoch()
            self.scheduler.step()
            val_loss, val_acc = self.eval_language_classifier_epoch()
            epoch_bar.set_postfix(
                loss=f'{train_loss:.4f}',
                val_acc=f'{val_acc:.4f}',
                refresh=False,
            )
            print_if_verbose(
                self.verbose,
                f'{"-" * 50} language classification '
                f'train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f} '
                f'{"-" * 50}'
            )
        return train_loss, val_loss, val_acc
    
    @staticmethod
    def nullspace_projection(w: torch.Tensor) -> torch.Tensor:
        """
        P = I - w w^T / ||w||^2.
        w: (d,) or (d, 1)  ->  P: (d, d)
        """
        w = w.float().contiguous().view(-1, 1)  # (d, 1)
        d = w.shape[0]
        norm_sq = (w.T @ w).clamp_min(1e-12)
        return torch.eye(d, dtype=w.dtype, device=w.device) - (w @ w.T) / norm_sq

    def chance_accuracy(self) -> float:
        """Majority-class accuracy (the INLP convergence target)."""
        y = self.y.clone()
        return max((y == 1).float().mean().item(), (y == 0).float().mean().item())

    def _init_projection(self) -> torch.Tensor:
        d = self.H.shape[1]
        P_perp = torch.eye(d, dtype=torch.float32)  # keep on CPU; d is small
        # Do not snapshot full H (n can be 1e5+); only track projection matrices.
        self.H_history = []
        self.P_history = [P_perp.clone()]
        return P_perp

    def remove_direction(
        self,
        w: torch.Tensor,
        H_cur: torch.Tensor,
        P_perp: torch.Tensor,
        chunk_size: int = 4096,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rank-1 null-space projection along w and accumulate into P_perp.

        H is projected in chunks on self.device to avoid holding the full (n, d)
        matrix on GPU; P_perp (d, d) is updated on device then moved back to CPU.
        """
        P_t = self.nullspace_projection(w).to(self.device)
        n_chunks = (H_cur.shape[0] + chunk_size - 1) // chunk_size
        pieces: list[torch.Tensor] = []
        chunk_iter = range(0, H_cur.shape[0], chunk_size)
        # if n_chunks > 4:
        #     chunk_iter = tqdm(
        #         chunk_iter,
        #         total=n_chunks,
        #         desc='project H',
        #         leave=False,
        #         dynamic_ncols=True,
        #     )
        # else:
        for i in chunk_iter:
            chunk = H_cur[i : i + chunk_size].to(self.device, non_blocking=True)
            pieces.append((chunk @ P_t).cpu())
            del chunk
            gc.collect()
            torch.cuda.empty_cache()

        H_new = torch.cat(pieces, dim=0)
        P_perp_new = (P_perp.to(self.device) @ P_t).cpu()
        del P_t, pieces
        gc.collect()
        torch.cuda.empty_cache()

        return H_new, P_perp_new


    #####   metrics for capability probing #####
    @staticmethod
    def probe_capability(
        H: torch.Tensor,
        rewards: torch.Tensor,
        *,
        epochs: int = 50,
        batch_size: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        split_ratio: float = 0.95,
        seed: int = 42,
        desc: str = 'cap probe',
    ) -> float:
        """
        BCEWithLogits on reward ∈ {0,1}, AdamW + cosine, return final val accuracy.
        """

        H = H.float().contiguous().view(H.shape[0], -1)
        rewards = rewards.float().contiguous().view(-1, 1)
        assert H.shape[0] == rewards.shape[0], (H.shape, rewards.shape)

        n = H.shape[0]
        train_size = round(n * split_ratio)
        train_indices = list(range(train_size))
        test_indices = list(range(train_size, n))

        dataset = BinaryLinearDataset(H, rewards)
        dataloader_train = DataLoader(
            Subset(dataset, train_indices),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            generator=self.generator,
        )
        dataloader_test = DataLoader(
            Subset(dataset, test_indices),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model = BinaryLinearClassifier(input_dim=H.shape[1]).to(self.device)
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)

        # build loss function
        rewards_train = rewards[train_indices].view(-1)
        n_pos = (rewards_train == 1).sum().clamp(min=1).float()
        n_neg = (rewards_train == 0).sum().clamp(min=1).float()
        pos_weight = (n_neg / n_pos).to(self.device).view(1)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        val_acc = 0.0
        epoch_bar = tqdm(range(epochs), desc=desc, leave=False, dynamic_ncols=True)
        for epoch in epoch_bar:
            model.train()
            for batch_hs, batch_y in dataloader_train:
                batch_hs = batch_hs.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                logits = model(batch_hs)
                loss = loss_fn(logits, batch_y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for batch_hs, batch_y in dataloader_test:
                    batch_hs = batch_hs.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    logits = model(batch_hs)
                    # trainer / scan_layers: threshold logit at 0 (== prob >= 0.5)
                    correct += ((logits >= 0).float() == batch_y).sum().item()
                    total += batch_y.shape[0]
            val_acc = correct / max(total, 1)
            epoch_bar.set_postfix(val_acc=f'{val_acc:.4f}', refresh=False)
        return val_acc


    def inlp(self):
        # Keep large H on CPU; only classifier / P live on self.device.
        self.H = self.H.cpu().contiguous().view(self.H.shape[0], -1)
        self.y = self.y.cpu().contiguous().view(-1)
        chance_acc = self.chance_accuracy()
        P_perp = self._init_projection()
        d = self.H.shape[1]

        P_lang_rows: list[torch.Tensor] = []
        accs: list[float] = []
        H_cur = self.H  # projected copies are allocated inside remove_direction

        iter_bar = tqdm(
            range(1, self.T + 1),
            desc='INLP iters',
            leave=False,
            dynamic_ncols=True,
        )
        for t in iter_bar:
            # Train on the *current* (already projected) representations.
            self._rebuild_for_H(H_cur)
            _, _, acc = self.train_language_classifier(desc=f'epochs[iter {t}/{self.T}]')
            accs.append(acc)
            iter_bar.set_postfix(
                acc=f'{acc:.4f}',
                chance=f'{chance_acc:.4f}',
                removed=len(P_lang_rows),
                refresh=False,
            )

            if acc <= chance_acc + self.chance_tolerance:
                iter_bar.set_postfix(
                    acc=f'{acc:.4f}',
                    status='converged',
                    removed=len(P_lang_rows),
                    refresh=True,
                )
                print_if_verbose(
                    self.verbose,
                    f'[INLP] iter {t}/{self.T} | clf_acc={acc:.4f} '
                    f'(chance={chance_acc:.4f}) -> converged, stop'
                )
                break

            w = self.classifier.query_weight().float().cpu()  # (d,)
            H_cur, P_perp = self.remove_direction(w, H_cur, P_perp)
            self.P_history.append(P_perp.clone())
            P_lang_rows.append(w.detach().cpu())
            print_if_verbose(
                self.verbose,
                f'[INLP] iter {t}/{self.T} | clf_acc={acc:.4f} '
                f'(chance={chance_acc:.4f}) -> remove direction d={w.numel()}'
            )

        P_lang = (
            torch.stack(P_lang_rows, dim=0)
            if P_lang_rows
            else torch.empty((0, d), dtype=torch.float32)
        )
        return H_cur, P_perp.cpu(), P_lang, accs

    def fresh_language_classifier_accuracy(self, H_proj: torch.Tensor) -> float:
        """Train a fresh linear probe on projected H; success => near chance."""
        self._rebuild_for_H(H_proj)
        _, _, acc = self.train_language_classifier(desc='fresh probe epochs')
        return acc

    def run(self):
        H_proj, P_perp, P_lang, accs = self.inlp()

        chance_acc = self.chance_accuracy()
        print(f'\n[INLP-run] iters run: {len(accs)}')
        print(f'[INLP-run] acc per iter: {[round(a, 4) for a in accs]}')
        print(f'[INLP-run] removed directions: {P_lang.shape[0]} (d={H_proj.shape[1]})')

        acc_after = self.fresh_language_classifier_accuracy(H_proj)
        print_if_verbose(
            self.verbose,
            f'[INLP-run] fresh language classifier acc after INLP: {acc_after:.4f} '
            f'(chance={chance_acc:.4f})'
        )
        return H_proj, P_perp, P_lang, accs, acc_after


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_layer_indices(arg: str, layer_keys: list[int]) -> list[int]:
    """Resolve --layer_indices: 'all' | '1,2,3' | '5' against available keys."""
    arg = arg.strip()
    keys = sorted(layer_keys)
    if arg == 'all':
        return keys
    if ',' in arg:
        wanted = sorted(int(x) for x in arg.split(','))
    else:
        wanted = [int(arg)]
    missing = [i for i in wanted if i not in keys]
    if missing:
        print(f'layer_indices not in hiddens: {missing}')
        raise KeyError(f'layer_indices not in hiddens: {missing}')
    return wanted


def load_hiddens(path: str | Path) -> dict[int, torch.Tensor]:
    obj = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    assert isinstance(obj, dict)
    hs = {int(k):v for k,v in obj.items()} # layer_idx: (N, hidden_dim)
    return hs
    

def discover_layer_keys_and_counts(
    hs_dir: Path,
    langs: list[str],
    template: str,
) -> tuple[list[int], dict[str, int], dict[str, Path]]:
    """Peek each lang file for layer keys + N without merging tensors."""
    paths: dict[str, Path] = {} # lang -> path
    counts: dict[str, int] = {} # lang -> N
    key_sets: list[set[int]] = [] # layer keys
    for lang in langs:
        assert lang in LANG2ID, f'unknown lang={lang}; known={list(LANG2ID)}'
        path = Path(hs_dir) / template.format(lang=lang)
        assert path.exists(), f'missing hs for lang={lang}: {path}'
        paths[lang] = path
        obj = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        key_sets.append({int(k) for k in obj.keys()}) # layer keys
        any_H = next(iter(obj.values())) # get any H tensor
        n = int(any_H.shape[0]) # n is the number of samples
        counts[lang] = n
        del obj
        gc.collect()
    common_layers = sorted(set.intersection(*key_sets))
    print(f'[data] langs={langs} counts={counts} common_layers={len(common_layers)}')
    return common_layers, counts, paths


def load_layer_multilingual(
    handles: dict[str, dict],
    langs: list[str],
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull one layer from already-opened (mmap) lang handles; concat H + lang_ids."""
    chunks: list[torch.Tensor] = [] # for storing hidden states
    id_chunks: list[torch.Tensor] = [] # for storing language ids (one-hot encoding)
    for lang in langs:
        obj = handles[lang]
        assert layer_idx in obj or str(layer_idx) in obj
        H = obj[layer_idx] if layer_idx in obj else obj[str(layer_idx)]
        # clone so we own a dense tensor independent of the mmap handle
        H = H.float().contiguous().clone()
        chunks.append(H)
        id_chunks.append(torch.full((H.shape[0],), LANG2ID[lang], dtype=torch.long))
    H_all = torch.cat(chunks, dim=0)
    lang_ids = torch.cat(id_chunks, dim=0)
    del chunks, id_chunks
    return H_all, lang_ids


def load_multilingual_rewards(
    reward_dir: str | Path,
    langs: list[str],
    template: str,
    expected_counts: dict[str, int] | None = None,
) -> torch.Tensor:
    """Concatenate per-language reward tensors in the same lang order as H.
       For probing res acc.
    """
    reward_dir = Path(reward_dir)
    chunks: list[torch.Tensor] = []
    for lang in langs:
        name = template.format(lang=lang)
        path = reward_dir / name
        assert path.exists(), f'missing reward for lang={lang}: {path}'
        reward_single = torch.load(path, map_location='cpu', weights_only=True).float().contiguous().view(-1) # (N,)
        if expected_counts is not None and lang in expected_counts:
            assert len(reward_single) == expected_counts[lang], f'reward N mismatch for {lang}: got {len(reward_single)}, expected hs N={expected_counts[lang]} ({path})'
        print(f'[data] reward {lang}: {path.name}  N={len(reward_single)}  '
              f'mean={reward_single.mean().item():.4f}')
        chunks.append(reward_single)
    rewards = torch.cat(chunks, dim=0)
    print(f'[data] merged rewards N={len(rewards)} mean={rewards.mean().item():.4f}')
    return rewards


def set_args():
    p = argparse.ArgumentParser(
        description='Run multilingual INLP (one-vs-rest) on pooled hidden states.'
    )
    p.add_argument(
        '--hs_dir',
        type=str,
        default='/root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct',
        help='directory containing per-language hs_*.pt files.',
    )
    p.add_argument(
        '--langs',
        type=str,
        default='en,zh',
        help='comma-separated languages to pool, e.g. en,zh or en,zh,es,vi,tr.',
    )
    # p.add_argument(
    #     '--split',
    #     type=str,
    #     default='train',
    #     choices=['train', 'test'],
    #     help='which split of hiddens to fit INLP on.',
    # )
     p.add_argument(
        '--hs_template',
        type=str,
        default='hs_math_train_{lang}_n8_tokens1024.pt',
        help='filename template under --hs_dir; fields: {lang}.',
    )
    p.add_argument(
        '--reward_dir',
        type=str,
        default='/root/autodl-tmp/exp1_math/judge',
        help='directory containing per-language reward_*.pt files.',
    )
    p.add_argument(
        '--reward_template',
        type=str,
        default='reward_math_train_{lang}_n8_t1.5_tokens1024.pt',
        help='filename template under --reward_dir; fields: {lang}.',
    )
    p.add_argument(
        '--target_lang',
        type=str,
        default='all',
        help="'all' = one-vs-rest for every lang in --langs; "
             "or a single lang code like 'zh'.",
    )
    p.add_argument(
        '--layer_indices',
        type=str,
        default='all',
        help="'all' | '1,2,3' | '5' — layers to run INLP on.",
    )

    p.add_argument('--T', type=int, default=15, help='max INLP iterations.')
    p.add_argument(
        '--epochs_per_iter',
        type=int,
        default=30,
        help='epochs to train each per-iteration language classifier.',
    )
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument(
        '--chance_tolerance',
        type=float,
        default=0.02,
        help='stop when clf acc <= chance_acc + chance_tolerance.',
    )
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument(
        '--split_ratio',
        type=float,
        default=0.95,
        help='train fraction for the language classifier probe split.',
    )

    # capability probe (paper Δcap)
    p.add_argument(
        '--cap_epochs',
        type=int,
        default=50,
        help='epochs for capability (reward) probe before/after INLP.',
    )
    p.add_argument(
        '--cap_scope',
        type=str,
        default='target',
        choices=['target', 'all'],
        help="'target': probe reward only on target-lang samples (language-conditioned Δcap); "
             "'all': probe on the full multilingual pool.",
    )

    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--verbose', action='store_true')
    p.add_argument(
        '--save_H_proj',
        action='store_true',
        help='also save projected H (large: ~N*d floats per layer). default: off.',
    )
    p.add_argument(
        '--out_dir',
        type=str,
        default='/root/autodl-tmp/exp1_math/inlp/Qwen2.5-3B-Instruct',
    )
    return p.parse_args()


def run_single_layer(
    H: torch.Tensor,
    y_labels: torch.Tensor,
    rewards: torch.Tensor,
    lang_ids: torch.Tensor,
    target_lang: str,
    layer_idx: int,
    args: argparse.Namespace,
    device: torch.device,
    chance_acc: float,
    n_pos: int,
    n_neg: int,
) -> dict:
    out_dir = Path(args.out_dir) / f'target_{target_lang}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f'\n----- target={target_lang} layer={layer_idx} | ' 
        f'N={len(H)} d={H.shape[1]} | n_pos={n_pos} n_neg={n_neg} -----'
    )
    inlp_runner = INLP_Runner(
        H=H,
        y=y_labels.clone(),
        T=args.T,
        epochs_per_iter=args.epochs_per_iter,
        lr=args.lr,
        weight_decay=args.weight_decay,
        chance_tolerance=args.chance_tolerance,
        batch_size=args.batch_size,
        verbose=args.verbose,
        split_ratio=args.split_ratio,
        seed=args.seed,
        device=device,
    )
    H_proj, P_perp, P_lang, accs, acc_after = inlp_runner.run()

    # --- capability probe (paper Sec 4.2 / Eq. 3 Δcap) ---
    cap_acc_before = float('nan')
    cap_acc_after = float('nan')
    delta_cap = float('nan')
    if args.cap_scope == 'target':
        mask = lang_ids == LANG2ID[target_lang]
        H_cap = H[mask]
        H_cap_proj = H_proj[mask]
        r_cap = rewards[mask]
        scope_tag = f'target={target_lang}'
    else:
        H_cap, H_cap_proj, r_cap = H, H_proj, rewards
        scope_tag = 'all'
    print(
        f'[cap] scope={scope_tag} N={len(r_cap)} '
        f'reward_mean={r_cap.float().mean().item():.4f}'
    )
    cap_acc_before = probe_capability(
        H_cap,r_cap,
        epochs=args.cap_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        split_ratio=args.split_ratio,
        seed=args.seed,
        desc=f'cap before L{layer_idx}',
    )
    cap_acc_after = probe_capability(
        H_cap_proj,r_cap,
        epochs=args.cap_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        split_ratio=args.split_ratio,
        seed=args.seed,
        desc=f'cap after L{layer_idx}'
    )
    delta_cap = cap_acc_after - cap_acc_before
    print(
        f'[cap] before={cap_acc_before:.4f} after={cap_acc_after:.4f} '
        f'Δcap={delta_cap:+.4f}'
    )
    
    del H_cap, H_cap_proj, r_cap

    if args.save_H_proj:
        torch.save(H_proj, out_dir / f'H_proj_layer{layer_idx}.pt')
    torch.save(P_perp, out_dir / f'P_perp_layer{layer_idx}.pt')
    torch.save(P_lang, out_dir / f'P_lang_layer{layer_idx}.pt')
    torch.save(
        torch.tensor(accs, dtype=torch.float32),
        out_dir / f'accs_layer{layer_idx}.pt',
    )

    del inlp_runner, H_proj
    gc.collect()
    torch.cuda.empty_cache()

    metrics = {
        'target_lang': target_lang,
        'layer': layer_idx,
        'n_iters': len(accs),
        'n_removed': int(P_lang.shape[0]),
        'd': int(H.shape[1]),
        'acc_first': accs[0] if accs else float('nan'),
        'acc_last': accs[-1] if accs else float('nan'),
        'acc_after': acc_after,
        'chance_acc': chance_acc,
        'n_pos': n_pos,
        'n_neg': n_neg,
        'cap_acc_before': cap_acc_before,
        'cap_acc_after': cap_acc_after,
        'delta_cap': delta_cap,
        'cap_scope': args.cap_scope,
    }
    return metrics


def main():
    args = set_args()

    langs = [x.strip() for x in args.langs.split(',') if x.strip()]

    # for all target languages
    if args.target_lang.strip() == 'all':
        targets = langs
    else: # for single target language
        assert ',' not in args.target_lang.strip() and args.target_lang.strip() in langs, target_lang
        targets = [args.target_lang.strip()]

    hs_dir = Path(args.hs_dir)
    # get common layers and paths for build handles
    common_layers, counts, paths = discover_layer_keys_and_counts(
        hs_dir, langs, args.hs_template,
    )
    layer_indices = parse_layer_indices(args.layer_indices, layer_keys=common_layers)

    # load rewards for capbility probing
    rewards = load_multilingual_rewards(
        reward_dir=args.reward_dir,
        langs=langs,
        template=args.reward_template,
        expected_counts=counts,
    )

    print('[data] opening mmap handles ...')
    # build handles for efficient loading of hidden states for language classification
    handles = {
        lang: torch.load(paths[lang], map_location='cpu', weights_only=True, mmap=True)
        for lang in langs
    }

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    # prepare per-target summary files lazily
    per_target_rows: dict[str, list[dict]] = {t: [] for t in targets} # for storing results for each target language
    all_rows: list[dict] = [] # for storing results for all target languages

    layer_bar = tqdm(layer_indices, desc='INLP over layers', dynamic_ncols=True)
    for layer_idx in layer_bar:
        H, lang_ids = load_layer_multilingual(handles, langs, layer_idx)
        assert rewards is not None and len(H) == len(rewards), \
        f'layer {layer_idx}: H N={len(H)} != rewards N={len(rewards)}' # for checking
        for target in targets:
            layer_bar.set_postfix(layer=layer_idx, target=target, refresh=True)
            target_id = LANG2ID[target]
            y_labels = (lang_ids == target_id).long() # build 0-1 labels for language classification
            n_pos, n_neg = int((y_labels == 1).sum()), int((y_labels == 0).sum())
            chance_acc = max(n_pos, n_neg) / len(y_labels) # for chance accuracy, as baseline
            row = run_single_layer(
                H=H,
                y_labels=y_labels,
                rewards=rewards,
                lang_ids=lang_ids,
                target_lang=target,
                layer_idx=layer_idx,
                args=args,
                device=device,
                chance_acc=chance_acc,
                n_pos=n_pos,
                n_neg=n_neg,
            )
            per_target_rows[target].append(row)
            all_rows.append(row)
        del H, lang_ids
        gc.collect()
        torch.cuda.empty_cache()

    for target, rows in per_target_rows.items():
        summary_path = Path(args.out_dir) / f'target_{target}' / 'summary.csv'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f'[INLP] target={target} summary -> {summary_path}')

    summary_path = Path(args.out_dir) / 'summary_all.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\n[INLP] all targets done. summary -> {summary_path}')
    print(f'[INLP] counts={counts}')

    print('all done!')


if __name__ == '__main__':
    main()

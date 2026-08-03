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
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, random_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu': print('using cpu')


class BinaryLinearClassifier(nn.Module):
    """Single-layer logistic regression: logit = H @ w + b (w is the protected direction)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1) # weight shape: d,1, bias: 1
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_normal_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: n,d
        # return self.linear(H).squeeze(-1)  # (n,1) -> (n,)
        return self.linear(H) # (n,1)

    def weight_vector(self) -> torch.Tensor:
        # return self.linear.weight.detach().squeeze(0)  # (d,)
        return self.linear.weight.detach() # (d,1)
    
    def count_params(self) -> int:
        return self.linear.weight.numel() + self.linear.bias.numel()


class BinaryLinearDataset(Dataset):
    def __init__(
        self,
        hs:torch.Tensor,
        y:torch.Tensor,
    ):
        super(BinaryLinearDataset,self).__init__()
        self.hs = hs.contiguous().view(hs.shape[0],-1)
        self.y = y.contiguous().view(-1,1)
        assert self.hs.shape[0] == self.y.shape[0]
    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self,idx:int):
        return self.hs[idx],self.y[idx]
    

def collate_fn(batch:list[tuple[torch.Tensor,torch.Tensor]]):
    batch_hs,batch_y=[],[]
    for item in batch:
        batch_hs.append(item[0])
        batch_y.append(item[1])
    return torch.stack(batch_hs,dim=0),torch.stack(batch_y,dim=0)




class Runner:
    def __init__(
        self,
        H:torch.Tensor,
        y:torch.Tensor,
        T:int=30,
        epochs_per_iter:int=50,
        lr:float=1e-3,
        weight_decay:float=0.0,
        chance_tolerance:float=0.02, # stop when the classifier accuracy falls to chance + chance_tolerance
        batch_size:int=64,
        verbose:bool=True,
        split_ratio:float=0.95,
    ):
        self.H = H
        self.y = y
        self.T = T
        self.epochs_per_iter = epochs_per_iter
        self.lr = lr
        self.weight_decay = weight_decay
        self.chance_tolerance = chance_tolerance
        self.batch_size = batch_size
        self.verbose = verbose
        self.split_ratio = split_ratio

        self.train_size = round(len(y) * split_ratio)
        self.test_size = len(y) - self.train_size

        self.generator = torch.Generator().manual_seed(42)
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu': print('using cpu')

        self.H_history = []
        self.P_history = []

        self.build_classification_dataloader()
        self.build_classifier()
        self.build_optimizer()
        self.build_scheduler()
        self.build_loss_fn()


    def build_classification_dataloader(self):
        self.dataset = BinaryLinearDataset(self.H,self.y)
        self.dataset_train,self.dataset_test = random_split(
            self.dataset,
            [self.train_size,self.test_size],
            generator=self.generator
        )
        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            generator=self.generator,
        )
        self.dataloader_test = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            generator=self.generator,
        )

    def build_classifier(self):
        self.classifier = BinaryLinearClassifier(
            input_dim=self.H.shape[1],
        ).to(self.device)
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

    def build_loss_fn(self):
        self.n_pos = (self.y == 1).sum().clamp(min=1)
        self.n_neg = (self.y == 0).sum().clamp(min=1)
        self.pos_weight = torch.tensor([self.n_neg / self.n_pos],device=self.device)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
    
    def train_classifier_epoch(self):
        self.classifier.train()
        epoch_loss = 0.0
        bar_train = tqdm(self.dataloader_train,desc='Training Classifier')
        for batch_hs, batch_y in bar_train:
            batch_hs = batch_hs.to(self.device)
            batch_y = batch_y.to(self.device)
            logits = self.classifier(batch_hs)
            loss = self.loss_fn(logits, batch_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        bar_train.set_postfix(loss=epoch_loss / len(self.dataloader_train))
        return epoch_loss / len(self.dataloader_train)
    
    def train_classifier(self):
        for epoch in range(self.epochs_per_iter):
            train_loss = self.train_classifier_epoch()
            self.scheduler.step()
            val_loss, val_acc = self.eval_classifier_epoch()
            if self.verbose:
                print(
                    '-'*50,
                    f'train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}',
                    '-'*50,
                )
        return train_loss, val_loss, val_acc
    
    def eval_classifier_epoch(self):
        self.classifier.eval()
        epoch_acc = 0.0
        epoch_loss = 0.0
        for batch_hs, batch_y in self.dataloader_test:
            batch_hs = batch_hs.to(self.device)
            batch_y = batch_y.to(self.device)
            with torch.no_grad():
                logits = self.classifier(batch_hs)
                loss = self.loss_fn(logits, batch_y)
                acc = ((torch.sigmoid(logits) >= 0.5).float() == batch_y).float().mean().item()
                epoch_acc += acc
                epoch_loss += loss.item()
        epoch_acc /= len(self.dataloader_test)
        epoch_loss /= len(self.dataloader_test)
        return epoch_loss, epoch_acc

    @staticmethod
    def nullspace_projection(w: torch.Tensor) -> torch.Tensor:
        """
        P = I - w w^T / ||w||^2, where w is the weight vector of the classifier.
        Rank-1 projection onto the subspace orthogonal to the direction w.
        w: (d,)
        P: (d, d)
        where d is the dimension of the hidden states.
        """
        w = w.float().contiguous().view(-1,1)
        d = w.shape[1]
        norm_sq = (w.T @ w).clamp_min(1e-12)
        return torch.eye(d, dtype=w.dtype, device=w.device) - w @ w.T / norm_sq


    # ------------------------------------------------------------------
    # For INLP below
    # ------------------------------------------------------------------
    def chance_accuracy(self) -> float:
        """Majority-class accuracy (the INLP convergence target)."""
        return max(
            (self.y == 1).float().mean().item(),
            (self.y == 0).float().mean().item(),
        )

    def _init_projection(self) -> torch.Tensor:
        """Reset histories and return the identity projection P_perp = I_d."""
        d = self.H.shape[1] # d
        P_perp = torch.eye(d, device=self.device, dtype=self.H.dtype)
        self.H_history = [self.H.clone().detach().cpu()]
        self.P_history = [P_perp.clone().detach().cpu()]
        return P_perp

    def remove_direction(
        self, w: torch.Tensor, H_cur: torch.Tensor, P_perp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the rank-1 null-space projection along w and accumulate it."""
        P_t = self.nullspace_projection(w).to(self.device)
        return H_cur @ P_t, P_perp @ P_t


    def inlp(self):
        self.H = self.H.to(self.device).contiguous().view(self.H.shape[0], -1) # (n, d)
        self.y = self.y.to(self.device).contiguous().view(-1, 1) # (n, 1)
        chance_acc = self.chance_accuracy()
        P_perp = self._init_projection()
        d = self.H.shape[1] # d

        P_lang_rows: list[torch.Tensor] = []
        accs: list[float] = []
        H_cur = self.H.clone()

        for t in range(1, self.T + 1):
            # 1. train a linear classifier on the current (already-projected) H
            _, _, acc = self.train_classifier()
            accs.append(acc)

            # 2. stop if the classifier is already at chance + chance_tolerance (no more linear signal)
            if acc <= chance_acc + self.chance_tolerance:
                if self.verbose:
                    print(f'[INLP] iter {t}/{self.T} | clf_acc={acc:.4f} (chance={chance_acc:.4f}) '
                          f'-> converged, stop')
                break

            # 3. record state, then remove the classifier's weight direction
            self.H_history.append(H_cur.clone().detach().cpu())
            self.P_history.append(P_perp.clone().detach().cpu())
            w = self.classifier.weight_vector().to(self.device)
            H_cur, P_perp = self.remove_direction(w, H_cur, P_perp)
            P_lang_rows.append(w.detach().cpu())
            if self.verbose:
                print(f'[INLP] iter {t}/{self.T} | clf_acc={acc:.4f} (chance={chance_acc:.4f}) '
                      f'-> remove direction {w.shape[0]}')

        P_lang = torch.stack(P_lang_rows, dim=0) if P_lang_rows else torch.empty((0, d)) # (k, d)
        return H_cur, P_perp, P_lang, accs # (n, d), (d, d), (k, d), list[float]


    def fresh_probe_accuracy(self, H_proj: torch.Tensor) -> float:
        """Train a fresh linear probe on the projected H to verify language info is gone.

        Reuses the build_* pipeline on the projected representations and returns the
        final validation accuracy. A successful INLP run drives this down to chance.
        """
        self.H = H_proj
        self.build_classification_dataloader()
        self.build_classifier()
        self.build_optimizer()
        self.build_scheduler()
        _, _, acc = self.train_classifier()
        return acc

    def run(self):
        # 1. remove language-predictive directions via INLP (paper Sec. 3.2, Eq. 2)
        H_proj, P_perp, P_lang, accs = self.inlp()

        # 2. summary of the removal trajectory
        chance_acc = self.chance_accuracy()
        print(f'\n[INLP-run] iters run: {len(accs)}')
        print(f'[INLP-run] acc per iter: {[round(a, 4) for a in accs]}')
        print(f'[INLP-run] removed directions: {P_lang.shape[0]} (d={self.H.shape[1]})')

        # 3. fresh language probe on the projected H — confirms language info is gone
        #    (paper Sec. 4.2: drop in language classification accuracy after INLP)
        acc_after = self.fresh_probe_accuracy(H_proj)
        print(f'[INLP-run] fresh clf acc after INLP: {acc_after:.4f} '
              f'(chance={chance_acc:.4f})')

        return H_proj, P_perp, P_lang, accs, acc_after
    
    


def parse_layer_indices(arg: str, n_layers: int) -> list[int]:
    """Resolve --layer_indices: 'all' | '1,2,3' | '5' -> sorted list of layer indices."""
    arg = arg.strip()
    if arg == 'all':
        return list(range(n_layers))
    if ',' in arg:
        return sorted(int(x) for x in arg.split(','))
    return [int(arg)]


def load_hiddens(path: str | Path) -> dict[int, torch.Tensor]:
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(obj, dict):
        return {int(k): v.float() for k, v in obj.items()}
    else:
        raise ValueError


def load_labels(path: str | Path) -> torch.Tensor:
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(obj, torch.Tensor):
        return obj.long().view(-1)
    else:
        raise ValueError


def set_args():
    p = argparse.ArgumentParser(
        description='Run INLP to remove language-predictive directions from hidden states.'
    )
    # data
    p.add_argument('--hiddens_path', type=str, required=True,
                   help='path to dict[layer_idx -> Tensor(N, d)] of hidden states.')
    p.add_argument('--label_path', type=str, required=True,
                   help='path to Tensor(N,) of integer language ids (one-vs-rest is built '
                        'from --target_lang_id).')
    p.add_argument('--target_lang_id', type=int, required=True,
                   help='language id treated as the positive class (one-vs-rest).')
    p.add_argument('--layer_indices', type=str, default='all',
                   help="'all' | '1,2,3' | '5' — layers to run INLP on.")

    # INLP hyperparameters
    p.add_argument('--T', type=int, default=30, help='max INLP iterations.')
    p.add_argument('--epochs_per_iter', type=int, default=50,
                   help='epochs to train each per-iteration language classifier.')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--chance_tolerance', type=float, default=0.02,
                   help='stop when clf acc <= chance_acc + chance_tolerance.')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--split_ratio', type=float, default=0.95,
                   help='train fraction of the train/test split for the language classifier.')

    # runtime
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--out_dir', type=str, default='./inlp_results')
    return p.parse_args()


def main():
    args = set_args()
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu': print('using cpu')
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = load_hiddens(args.hiddens_path)
    lang_ids = load_labels(args.label_path)
    layer_indices = parse_layer_indices(args.layer_indices, n_layers=len(layers))

    # one-vs-rest binary labels for the target language
    y_ovr = (lang_ids == args.target_lang_id).long()
    n_pos = int((y_ovr == 1).sum())
    n_neg = int((y_ovr == 0).sum())
    print(f'[INLP] target_lang_id={args.target_lang_id} | '
          f'n_pos={n_pos} n_neg={n_neg} | chance_acc={max(n_pos, n_neg) / len(y_ovr):.4f}')

    summary_rows: list[dict] = []
    bar = tqdm(layer_indices, desc='INLP over layers')
    for layer_idx in bar:
        H = layers[layer_idx]
        assert len(H) == len(y_ovr), \
            f'layer {layer_idx}: hiddens {len(H)} != labels {len(y_ovr)}'

        runner = Runner(
            H=H, y=y_ovr.clone(),
            T=args.T, epochs_per_iter=args.epochs_per_iter,
            lr=args.lr, weight_decay=args.weight_decay,
            chance_tolerance=args.chance_tolerance, batch_size=args.batch_size,
            verbose=args.verbose, split_ratio=args.split_ratio,
        )
        # Runner auto-detected device in __init__; honour --device if compatible.
        runner.device = device
        runner.classifier = runner.classifier.to(device)
        runner.pos_weight = runner.pos_weight.to(device)
        H_proj, P_perp, P_lang, accs, acc_after = runner.run()

        torch.save(H_proj, out_dir / f'H_proj_layer{layer_idx}.pt')
        torch.save(P_perp, out_dir / f'P_perp_layer{layer_idx}.pt')
        torch.save(P_lang, out_dir / f'P_lang_layer{layer_idx}.pt')
        torch.save(torch.tensor(accs, dtype=torch.float32),
                   out_dir / f'accs_layer{layer_idx}.pt')

        chance_acc = max(n_pos, n_neg) / len(y_ovr)
        summary_rows.append({
            'layer': layer_idx,
            'n_iters': len(accs),
            'n_removed': int(P_lang.shape[0]),
            'd': int(H.shape[1]),
            'acc_first': accs[0] if accs else float('nan'),
            'acc_last': accs[-1] if accs else float('nan'),
            'acc_after': acc_after,
            'chance_acc': chance_acc,
        })

    summary_path = out_dir / 'summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f'\n[INLP] done. summary -> {summary_path}')



if __name__ == '__main__':
    main()

"""
Iterative Null-space Projection (INLP) for removing language-predictive
directions, following the paper's formulation (Section 3.2, Eq. 2).

At each iteration i:
  1. Train a linear classifier w_i to predict a (binary) protected attribute
     from hidden states H (one-vs-rest for a given language).
  2. Project H onto the null space of w_i:
        P_t      = I - w_i w_i^T / ||w_i||^2          (rank-1, Eq. 2)
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

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

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
        chance_tol:float=0.02,
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
        self.chance_tol = chance_tol
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


    def inlp(self):
        self.H = self.H.to(self.device)
        self.H_history.append(self.H.clone().detach().cpu())
        self.y = self.y.to(self.device)
        n,d = self.H.shape
        chance_acc = max((self.y == 1).float().mean().item(), (self.y == 0).float().mean().item())
        P_perp = torch.eye(d, device=self.device, dtype=self.H.dtype)
        self.P_history.append(P_perp.clone().detach().cpu())
        P_lang_rows: list[torch.Tensor] = []
        accs: list[float] = []
        H_cur = self.H.clone()
        for t in range(1, self.T + 1):
            # 1. train a linear classifier on the current (already-projected) H
            clf, acc = self.train_classifier()
            accs.append(acc)
            if acc <= chance_acc + self.chance_tol:
                if self.verbose:
                    print(f'[INLP] iter {t}/{self.T} | clf_acc={acc:.4f} (chance={chance_acc:.4f}) '
                          f'-> converged, stop')
                break
            self.H_history.append(H_cur.clone().detach().cpu())
            self.P_history.append(P_perp.clone().detach().cpu())
            w = clf.weight_vector().to(self.device)
            P_t = self.nullspace_projection(w).to(self.device)
            H_cur = H_cur @ P_t
            P_perp = P_perp @ P_t
            P_lang_rows.append(w.detach().cpu())
            if self.verbose:
                print(f'[INLP] iter {t}/{self.T} | clf_acc={acc:.4f} (chance={chance_acc:.4f}) | '
                      f'removed {len(P_lang_rows)} dir(s)')
        P_lang = torch.stack(P_lang_rows, dim=0) if P_lang_rows else torch.empty((0, d))
        return H_cur, P_perp, P_lang, accs


    def run(self):
        pass
    
    







def inlp(
    H: torch.Tensor,
    y: torch.Tensor,
    T: int = 30,
    epochs_per_iter: int = 50,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    chance_tol: float = 0.02,
    batch_size: int = 64,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
    """
    Run INLP until the classifier falls to chance (or T iterations).

    Args:
        H: (n, d) hidden states at a single layer.
        y: (n,) binary protected labels (0/1), e.g. one-vs-rest for a language.
        T: max number of INLP iterations.
        epochs_per_iter: epochs to train each per-iteration classifier.
        lr / weight_decay: optimizer args for the per-iteration classifier.
        chance_tol: stop when clf acc <= chance_acc + chance_tol.
        verbose: print per-iteration accuracy.

    Returns:
        H_proj : (n, d) projected hidden states.
        P_perp : (d, d) cumulative projection P_1 @ ... @ P_t.
        P_lang : (k, d) stacked removed direction vectors w_i.
        accs   : list[float] per-iteration classifier accuracy.
    """
    H = H.to(device).float()
    y = y.to(device).long()

    n, d = H.shape
    chance_acc = max((y == 1).float().mean().item(), (y == 0).float().mean().item())

    P_perp = torch.eye(d, device=device, dtype=H.dtype)
    P_lang_rows: list[torch.Tensor] = []
    accs: list[float] = []

    H_cur = H.clone()
    for t in range(1, T + 1):
        # 1. train a linear classifier on the current (already-projected) H
        clf, acc = train_binary_classifier(
            H_cur, y, input_dim=d,
            epochs=epochs_per_iter, lr=lr, weight_decay=weight_decay, batch_size=batch_size
        )
        accs.append(acc)

        # 2. stop if the classifier is already at chance (no more linear signal)
        if acc <= chance_acc + chance_tol:
            if verbose:
                print(f'[INLP] iter {t}/{T} | clf_acc={acc:.4f} (chance={chance_acc:.4f}) '
                      f'-> converged, stop')
            break

        # 3. rank-1 null-space projection along the classifier's weight direction
        w = clf.weight_vector().to(device)        # (d,)
        P_t = nullspace_projection(w).to(device)   # (d, d)

        # 4. update H and accumulate cumulative projection
        H_cur = H_cur @ P_t
        P_perp = P_perp @ P_t
        P_lang_rows.append(w.detach().cpu())

        if verbose:
            print(f'[INLP] iter {t}/{T} | clf_acc={acc:.4f} (chance={chance_acc:.4f}) | '
                  f'removed {len(P_lang_rows)} dir(s)')

    P_lang = torch.stack(P_lang_rows, dim=0) if P_lang_rows else torch.empty((0, d))
    return H_cur, P_perp, P_lang, accs


    
    # def run_single(self):
    #     H_proj, P_perp, P_lang, accs = inlp(
    #         self.H,
    #         self.y,
    #         T=self.T,
    #         epochs_per_iter=self.epochs_per_iter,
    #         lr=self.lr,
    #         weight_decay=self.weight_decay,
    #         chance_tol=self.chance_tol,
    #         batch_size=self.batch_size,
    #         verbose=self.verbose,
    #     )
    #     print(f'\nINLP iters run: {len(accs)}')
    #     print(f'acc per iter: {[round(a, 4) for a in accs]}')
    #     print(f'removed directions: {P_lang.shape[0]} (d={self.H.shape[1]})')

    #     _, acc_after = train_binary_classifier(
    #         H_proj,
    #         self.y,
    #         input_dim=self.H.shape[1],
    #         epochs=self.epochs_per_iter,
    #         lr=self.lr,
    #         weight_decay=self.weight_decay,
    #     )
    #     print(f'fresh clf acc after INLP: {acc_after:.4f} (should be near chance=0.5)')
    #     return H_proj, P_perp, P_lang, accs, acc_after
    
    # def run(self):
    #     pass

    





def main():
    # # smoke test: inject a linear language signal, then erase it with INLP
    # torch.manual_seed(0)
    # n, d = 200, 64
    # H0 = torch.randn(n, d)
    # # binary protected attribute linearly readable from H
    # w_true = torch.randn(d)
    # y = (H0 @ w_true > 0).long()

    # H_proj, P_perp, P_lang, accs = inlp(
    #     H0, y, T=20, epochs_per_iter=80, lr=5e-2, verbose=True,
    # )

    # # a fresh classifier should fail to recover y after projection
    # _, acc_after = train_binary_classifier(
    #     H_proj, y, input_dim=d, epochs=80, lr=5e-2,
    # )
    # print(f'\nINLP iters run: {len(accs)}')
    # print(f'acc per iter: {[round(a, 4) for a in accs]}')
    # print(f'fresh clf acc after INLP: {acc_after:.4f} '
    #       f'(should be near chance=0.5)')
    # print(f'removed directions: {P_lang.shape[0]} (d={d})')
    pass



if __name__ == '__main__':
    main()

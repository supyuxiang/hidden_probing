"""
Iterative Null-space Projection (INLP) for removing language-predictive
directions, following the paper's formulation (Section 3.2, Eq. 2).

At each iteration i:
  1. Train a linear classifier w_i to predict a (binary) protected attribute
     from hidden states H (one-vs-rest for a given language).
  2. Project H onto the null space of w_i:
        P_t      = I - w_i w_i^T / ||w_i||^2          (rank-1, Eq. 2)
        H_{i+1}  = H_i @ P_t
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _BinaryLinearClassifier(nn.Module):
    """Single-layer logistic regression: logit = H @ w + b (w is the protected direction)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return self.linear(H).squeeze(-1)  # (n,)

    def weight_vector(self) -> torch.Tensor:
        return self.linear.weight.detach().squeeze(0)  # (d,)


def _train_binary_classifier(
    H: torch.Tensor,
    y: torch.Tensor,
    input_dim: int,
    epochs: int = 50,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    device: torch.device | str = 'cpu',
) -> tuple[_BinaryLinearClassifier, float]:
    """Train one binary linear classifier (BCE) on H -> y; return (model, train acc)."""
    H = H.to(device).float()
    y = y.to(device).float()

    model = _BinaryLinearClassifier(input_dim).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # class-balanced positive weight so the direction isn't dominated by the majority class
    n_pos = (y == 1).sum().clamp(min=1)
    n_neg = (y == 0).sum().clamp(min=1)
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for _ in range(epochs):
        logits = model(H)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        acc = ((torch.sigmoid(model(H)) >= 0.5).float() == y).float().mean().item()
    return model, acc


def nullspace_projection(w: torch.Tensor) -> torch.Tensor:
    """
    Eq. 2: P = I - w w^T / ||w||^2.
    Rank-1 projection onto the subspace orthogonal to the direction w.
    """
    w = w.float()
    d = w.shape[0]
    norm_sq = (w @ w).clamp_min(1e-12)
    return torch.eye(d, dtype=w.dtype, device=w.device) - torch.outer(w, w) / norm_sq


def one_vs_rest_labels(language_ids: torch.Tensor, target: int) -> torch.Tensor:
    """Build binary 0/1 labels: 1 where language_ids == target, else 0."""
    return (language_ids == target).long()


def inlp(
    H: torch.Tensor,
    y: torch.Tensor,
    T: int = 30,
    epochs_per_iter: int = 50,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    chance_tol: float = 0.02,
    device: torch.device | str | None = None,
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
        device: where to run. Defaults to H's device.
        verbose: print per-iteration accuracy.

    Returns:
        H_proj : (n, d) projected hidden states.
        P_perp : (d, d) cumulative projection P_1 @ ... @ P_t.
        P_lang : (k, d) stacked removed direction vectors w_i.
        accs   : list[float] per-iteration classifier accuracy.
    """
    if device is None:
        device = H.device
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
        clf, acc = _train_binary_classifier(
            H_cur, y, input_dim=d,
            epochs=epochs_per_iter, lr=lr, weight_decay=weight_decay, device=device,
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


def project(H: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Apply a (cumulative) projection matrix to hidden states: H @ P."""
    return H.float() @ P.to(H.device).float()


def main():
    # smoke test: inject a linear language signal, then erase it with INLP
    torch.manual_seed(0)
    n, d = 200, 64
    H0 = torch.randn(n, d)
    # binary protected attribute linearly readable from H
    w_true = torch.randn(d)
    y = (H0 @ w_true > 0).long()

    H_proj, P_perp, P_lang, accs = inlp(
        H0, y, T=20, epochs_per_iter=80, lr=5e-2, verbose=True,
    )

    # a fresh classifier should fail to recover y after projection
    _, acc_after = _train_binary_classifier(
        H_proj, y, input_dim=d, epochs=80, lr=5e-2,
    )
    print(f'\nINLP iters run: {len(accs)}')
    print(f'acc per iter: {[round(a, 4) for a in accs]}')
    print(f'fresh clf acc after INLP: {acc_after:.4f} '
          f'(should be near chance=0.5)')
    print(f'removed directions: {P_lang.shape[0]} (d={d})')


if __name__ == '__main__':
    main()

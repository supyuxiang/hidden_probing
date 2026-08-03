"""
Iterative Null-space Projection (INLP).

Reference: Ravfogel et al., 2020, "Null It Out: Guarding Protected Attributes
by Iterative Nullspace Projection".

Given hidden states H (n, d) and a protected attribute y (n,), we iteratively
train a linear classifier to predict y from H, then project H onto the null
space of the classifier's weight rowspace. After T iterations the cumulative
projection P_perp = P_1 @ P_2 @ ... @ P_T removes the linear information about y.

Key equations (matching the paper):
    P_t = I - W_t^T (W_t W_t^T)^+ W_t        (Eq. 5)
    H_{t+1} = H_t P_t                         (Eq. 6)
    P_perp = prod_{t=1}^{T} P_t               (Eq. 7)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exp1_math.model import Classifier_Linear


def _train_linear_classifier(
    H: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    input_dim: int,
    epochs: int = 50,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    device: torch.device | str = 'cpu',
) -> Classifier_Linear:
    """Train a single linear classifier (cross-entropy) on H -> y."""
    H = H.to(device).float()
    y = y.to(device).long()

    model = Classifier_Linear(input_dim=input_dim, output_dim=num_classes).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        logits = model(H)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def compute_projection(W: torch.Tensor) -> torch.Tensor:
    """
    Eq. 5: P = I - W^T (W W^T)^+ W.

    Args:
        W: (k, d) weight matrix of the linear classifier (rows = class directions).
    Returns:
        P: (d, d) projection matrix onto the null space of W's rowspace.
    """
    W = W.float()
    k, d = W.shape
    # (d, k) @ (k, k) @ (k, d) -> (d, d)
    WWT = W @ W.T                       # (k, k)
    WWT_pinv = torch.linalg.pinv(WWT)   # (k, k)
    P = torch.eye(d, dtype=W.dtype, device=W.device) - W.T @ WWT_pinv @ W
    return P


def inlp(
    H: torch.Tensor,
    y: torch.Tensor,
    T: int = 15,
    epochs_per_iter: int = 50,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """
    Run INLP for T iterations.

    Args:
        H: (n, d) hidden states.
        y: (n,) protected attribute labels (integers).
        T: number of INLP iterations.
        epochs_per_iter: epochs to train each per-iteration classifier.
        lr / weight_decay: optimizer args for the per-iteration classifier.
        device: where to run. Defaults to H's device.
        verbose: print per-iteration accuracy.

    Returns:
        H_proj: (n, d) projected hidden states after T iterations.
        P_perp: (d, d) cumulative projection matrix P_1 @ ... @ P_T.
        P_list: list of each P_t, for inspection.
    """
    if device is None:
        device = H.device
    H = H.to(device).float()
    y = y.to(device).long()

    n, d = H.shape
    num_classes = int(y.max().item()) + 1

    P_perp = torch.eye(d, device=device, dtype=H.dtype)
    P_list: list[torch.Tensor] = []
    H_cur = H.clone()

    for t in range(1, T + 1):
        # 1. train a linear classifier on the *current* (already-projected) H
        clf = _train_linear_classifier(
            H_cur, y, num_classes=num_classes, input_dim=d,
            epochs=epochs_per_iter, lr=lr, weight_decay=weight_decay, device=device,
        )

        # 2. accuracy of this classifier (information about y remaining in H)
        clf.eval()
        with torch.no_grad():
            acc = (clf(H_cur).argmax(dim=-1) == y).float().mean().item()

        # 3. projection matrix from the classifier's weight (rowspace removal)
        W = clf.linear.weight.detach()          # (num_classes, d)
        P_t = compute_projection(W).to(device)  # (d, d)

        # 4. update H and accumulate cumulative projection
        H_cur = H_cur @ P_t
        P_perp = P_perp @ P_t
        P_list.append(P_t.cpu())

        if verbose:
            print(f'[INLP] iter {t}/{T} | clf_acc={acc:.4f}')

    return H_cur, P_perp, P_list


def project(H: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Apply a (cumulative) projection matrix to hidden states: H @ P."""
    return H.float() @ P.to(H.device).float()


def set_args():
    import argparse
    p = argparse.ArgumentParser()

    return p.parse_args()

def main():
    # tiny smoke test
    torch.manual_seed(0)
    n, d, k = 200, 64, 3
    H0 = torch.randn(n, d)
    # inject a linear signal for a 3-class attribute
    W_true = torch.randn(k, d)
    y = (H0 @ W_true.T).argmax(dim=-1)

    H_proj, P_perp, _ = inlp(H0, y, T=10, epochs_per_iter=80, lr=5e-2, verbose=True)

    # a fresh linear classifier should fail to recover y after projection
    clf = _train_linear_classifier(H_proj, y, num_classes=k, input_dim=d,
                                   epochs=80, lr=5e-2)
    with torch.no_grad():
        acc_after = (clf(H_proj).argmax(dim=-1) == y).float().mean().item()
    print(f'accuracy after INLP: {acc_after:.4f} (should be near chance={1/k:.3f})')


if __name__ == '__main__':
    main()

"""
Scan all layers: train a linear probe (BCE) for each saved layer index and
report per-layer val accuracy averaged over all epochs.

Usage:
    CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/scan_layers.py \
        --hiddens_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math.pt \
        --reward_path  /root/autodl-tmp/exp1_math/judge/reward_math.pt \
        --num_epochs 50 \
        --batch_size 64 \
        --lr 1.0e-3 \
        --out_dir /root/autodl-tmp/exp1_math/scan
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exp1_math.model import Classifier_Linear, Classifier_MLP


def load_hiddens(path: str | Path) -> dict[int, torch.Tensor]:
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict):
        return {int(k): v.float() for k, v in obj.items()}
    # single tensor (N, d) -> treat as one layer at index 0
    return {0: obj.float()}


def load_rewards(path: str | Path) -> torch.Tensor:
    r = torch.load(path, map_location='cpu').float()
    return r.view(-1, 1)


def balanced_subsample(rewards: torch.Tensor, n_per_class: int, seed: int) -> torch.Tensor:
    """
    Pick `n_per_class` positive and `n_per_class` negative samples (by reward),
    shuffled. Returns the selected indices into the original array.

    n_per_class ==  0 -> use all samples (no subsampling)
    n_per_class == -1 -> auto: balance to the minority class count
                        (uses ALL of the minority class, subsamples the majority)
    n_per_class  >  0 -> that many per class
    """
    r = rewards.view(-1)
    if n_per_class == 0:
        return torch.arange(len(rewards))
    n_pos_total = int((r == 1).sum().item())
    n_neg_total = int((r == 0).sum().item())
    if n_per_class == -1:
        n_per_class = min(n_pos_total, n_neg_total)
    g = torch.Generator().manual_seed(seed)
    pos_idx = (r == 1).nonzero(as_tuple=True)[0]
    neg_idx = (r == 0).nonzero(as_tuple=True)[0]
    n_pos = min(n_per_class, len(pos_idx))
    n_neg = min(n_per_class, len(neg_idx))
    pos_sel = pos_idx[torch.randperm(len(pos_idx), generator=g)[:n_pos]]
    neg_sel = neg_idx[torch.randperm(len(neg_idx), generator=g)[:n_neg]]
    sel = torch.cat([pos_sel, neg_sel])
    sel = sel[torch.randperm(len(sel), generator=g)]
    return sel


def run_one_layer(
    hiddens: torch.Tensor,
    rewards: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
    val_ratio: float = 0.1,
) -> list[float]:
    """Train one probe for `epochs` epochs, return per-epoch val_acc list."""
    n = len(hiddens)
    train_n = round(n * (1 - val_ratio))
    val_n = n - train_n

    # fresh generator each layer -> identical train/val split across layers
    gen = torch.Generator().manual_seed(seed)
    ds = TensorDataset(hiddens, rewards)
    ds_train, ds_val = random_split(ds, [train_n, val_n], generator=gen)

    g_train = torch.Generator().manual_seed(seed)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, generator=g_train)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    model = Classifier_Linear(input_dim=hiddens.shape[1], output_dim=1).to(device)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=0.0)

    # pos_weight = n_neg / n_pos on the train split (balances the ~0.69/0.31 split)
    train_rw = rewards[ds_train.indices].float()
    n_pos = (train_rw == 1).sum().clamp(min=1)
    n_neg = (train_rw == 0).sum().clamp(min=1)
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    accs: list[float] = []
    for _ in range(epochs):
        model.train()
        for hs, rw in dl_train:
            hs = hs.to(device)
            rw = rw.to(device).float()
            logits = model(hs)
            loss = loss_fn(logits, rw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for hs, rw in dl_val:
                hs = hs.to(device)
                rw = rw.to(device)
                logits = model(hs)
                # threshold logit at 0 (== sigmoid >= 0.5)
                correct += ((logits >= 0).float() == rw).sum().item()
                total += rw.size(0)
        accs.append(correct / max(total, 1))
    return accs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hiddens_path', type=str, required=True)
    p.add_argument('--reward_path', type=str, required=True)
    p.add_argument('--num_epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1.0e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--n_per_class', type=int, default=-1,
                   help='balanced subsample: #pos and #neg to keep (total = 2x). '
                        '-1 = auto (balance to minority class count, max balanced set); '
                        '0 = use all; >0 = that many per class.')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--out_dir', type=str, default='./scan_results')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = load_hiddens(args.hiddens_path)
    rewards = load_rewards(args.reward_path)

    # sanity: every layer must align with rewards
    for k, v in layers.items():
        assert len(v) == len(rewards), f'layer {k}: hiddens {len(v)} != rewards {len(rewards)}'

    # balanced subsample (same indices across all layers -> fair comparison)
    sel = balanced_subsample(rewards, args.n_per_class, args.seed)
    rewards = rewards[sel]
    layers = {k: v[sel] for k, v in layers.items()}
    n_pos = int((rewards == 1).sum())
    n_neg = int((rewards == 0).sum())
    print(f'subsampled to {len(rewards)} (pos={n_pos}, neg={n_neg}) '
          f'-> majority baseline acc = {max(n_pos, n_neg) / len(rewards):.4f}')
    baseline_acc = max(n_pos, n_neg) / len(rewards)

    rows = []
    bar = tqdm(sorted(layers.keys()), desc='scanning layers')
    for layer_idx in bar:
        hs = layers[layer_idx]
        accs = run_one_layer(
            hs, rewards,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed,
            val_ratio=args.val_ratio,
        )
        accs_t = torch.tensor(accs)
        rows.append({
            'layer_idx': layer_idx,
            'mean_val_acc': accs_t.mean().item(),
            'best_val_acc': accs_t.max().item(),
            'final_val_acc': accs_t[-1],
            'std_val_acc': accs_t.std().item(),
        })
        bar.set_postfix(
            layer=layer_idx,
            mean_acc=f'{rows[-1]["mean_val_acc"]:.4f}',
            best_acc=f'{rows[-1]["best_val_acc"]:.4f}',
        )

    rows.sort(key=lambda r: r['layer_idx'])

    # ---- console table ----
    header = f'{"layer":>6} | {"mean_acc":>9} | {"best_acc":>9} | {"final_acc":>10} | {"std_acc":>8}'
    print('\n' + header)
    print('-' * len(header))
    for r in rows:
        print(f'{r["layer_idx"]:>6} | {r["mean_val_acc"]:>9.4f} | {r["best_val_acc"]:>9.4f} | '
              f'{r["final_val_acc"]:>10.4f} | {r["std_val_acc"]:>8.4f}')

    best = max(rows, key=lambda r: r['mean_val_acc'])
    print(f'\nbest layer by mean_val_acc: layer {best["layer_idx"]} '
          f'(mean={best["mean_val_acc"]:.4f}, best={best["best_val_acc"]:.4f})')

    # ---- CSV ----
    csv_path = out_dir / 'layer_scan.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['layer_idx', 'mean_val_acc', 'best_val_acc',
                                          'final_val_acc', 'std_val_acc'])
        w.writeheader()
        w.writerows(rows)
    print(f'csv saved to {csv_path}')

    # ---- Markdown ----
    md_path = out_dir / 'layer_scan.md'
    with open(md_path, 'w') as f:
        f.write('# Linear probing: per-layer val accuracy\n\n')
        f.write(f'hiddens: `{args.hiddens_path}`  \nrewards: `{args.reward_path}`  \n')
        f.write(f'epochs={args.num_epochs}, batch_size={args.batch_size}, lr={args.lr}, '
                f'seed={args.seed}, val_ratio={args.val_ratio}, '
                f'n_per_class={args.n_per_class} (used {len(rewards)} samples: '
                f'pos={n_pos}, neg={n_neg}, baseline_acc={baseline_acc:.4f})\n\n')
        f.write('| layer_idx | mean_val_acc | best_val_acc | final_val_acc | std_val_acc |\n')
        f.write('|---|---|---|---|---|\n')
        for r in rows:
            f.write(f'| {r["layer_idx"]} | {r["mean_val_acc"]:.4f} | {r["best_val_acc"]:.4f} '
                    f'| {r["final_val_acc"]:.4f} | {r["std_val_acc"]:.4f} |\n')
        f.write(f'\n**Best layer (by mean_val_acc): layer {best["layer_idx"]} '
                f'— mean={best["mean_val_acc"]:.4f}, best={best["best_val_acc"]:.4f}**\n')
    print(f'markdown saved to {md_path}')

    # ---- plot ----
    try:
        import matplotlib.pyplot as plt
        xs = [r['layer_idx'] for r in rows]
        plt.figure(figsize=(10, 5))
        plt.plot(xs, [r['mean_val_acc'] for r in rows], 'o-', label='mean_val_acc')
        plt.plot(xs, [r['best_val_acc'] for r in rows], 's--', alpha=0.6, label='best_val_acc')
        plt.axhline(baseline_acc, color='r', linestyle=':', label=f'majority baseline ({baseline_acc:.3f})')
        plt.xlabel('layer_idx')
        plt.ylabel('val accuracy')
        plt.title('Linear probing accuracy across layers')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fig_path = out_dir / 'layer_scan.png'
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f'plot saved to {fig_path}')
    except Exception as e:
        print(f'skipping plot: {e}')


if __name__ == '__main__':
    main()

"""
Publication-style plots for multilingual INLP results.

Reads outputs from inlp_runner.py:
  <out_dir>/summary_all.csv
  <out_dir>/target_<lang>/accs_layer{k}.pt
  <out_dir>/target_<lang>/P_lang_layer{k}.pt   (optional; for k check)

Produces (PNG + PDF under --fig_dir):
  01_layer_probe_accuracy.{png,pdf}
      Layer-wise language-probe accuracy: first INLP iter, last INLP iter,
      fresh probe after projection, and chance baseline. One panel per target lang.
  02_delta_lang_and_removed.{png,pdf}
      Δlang = acc_first − acc_after, and # removed directions k, vs layer.
  03_inlp_iteration_heatmap.{png,pdf}
      Per-iteration classifier accuracy (rows=layers, cols=INLP iters).
  04_inlp_iteration_curves.{png,pdf}
      Accuracy trajectories for a few representative layers.

Usage:
  python /root/hidden_prob/exp1_math/plot_inlp.py \\
      --inlp_dir /root/autodl-tmp/exp1_math/inlp/Qwen2.5-3B-Instruct/en_zh_train \\
      --fig_dir  /root/autodl-tmp/exp1_math/inlp/Qwen2.5-3B-Instruct/en_zh_train/figs
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Style — flat, print-friendly, no purple/glow AI defaults
# ---------------------------------------------------------------------------

LANG_COLORS = {
    'en': '#1B4F72',  # deep blue
    'zh': '#A93226',  # brick red
    'es': '#196F3D',  # forest
    'vi': '#B9770E',  # amber
    'tr': '#5B2C6F',  # plum
}

ACC_SERIES = {
    'acc_first': ('#1B4F72', 'o', 'first iter'),
    'acc_last': ('#5D6D7E', 's', 'last iter'),
    'acc_after': ('#A93226', 'D', 'fresh after INLP'),
    'chance_acc': ('#7F8C8D', '--', 'chance'),
}


def apply_style():
    mpl.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#2C3E50',
        'axes.labelcolor': '#1C2833',
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.color': '#1C2833',
        'ytick.color': '#1C2833',
        'font.size': 10,
        'font.family': 'DejaVu Sans',
        'legend.frameon': False,
        'legend.fontsize': 9,
        'grid.color': '#D5D8DC',
        'grid.linewidth': 0.6,
        'lines.linewidth': 1.8,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def lang_color(lang: str) -> str:
    return LANG_COLORS.get(lang, '#34495E')


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_summary(inlp_dir: Path) -> list[dict]:
    path = inlp_dir / 'summary_all.csv'
    if not path.exists():
        # fall back: concat per-target summaries
        rows: list[dict] = []
        for p in sorted(inlp_dir.glob('target_*/summary.csv')):
            with open(p, newline='') as f:
                rows.extend(csv.DictReader(f))
        if not rows:
            raise FileNotFoundError(
                f'no summary found under {inlp_dir}; run inlp_runner.py first'
            )
        return rows
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def as_float(row: dict, key: str) -> float:
    return float(row[key])


def as_int(row: dict, key: str) -> int:
    return int(float(row[key]))


def group_by_lang(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r['target_lang']].append(r)
    for lang in out:
        out[lang].sort(key=lambda r: as_int(r, 'layer'))
    return dict(out)


def load_accs_matrix(
    inlp_dir: Path, lang: str, layers: list[int],
) -> tuple[np.ndarray, int]:
    """
    Returns (A, T_max) where A[i, t] = clf acc at INLP iter t for layers[i].
    Missing / shorter trajectories are padded with NaN.
    """
    curves: list[np.ndarray] = []
    t_max = 0
    for layer in layers:
        path = inlp_dir / f'target_{lang}' / f'accs_layer{layer}.pt'
        if not path.exists():
            curves.append(np.array([], dtype=np.float64))
            continue
        a = torch.load(path, map_location='cpu', weights_only=True).float().numpy()
        curves.append(a)
        t_max = max(t_max, len(a))
    if t_max == 0:
        return np.full((len(layers), 1), np.nan), 0
    A = np.full((len(layers), t_max), np.nan, dtype=np.float64)
    for i, a in enumerate(curves):
        if len(a):
            A[i, : len(a)] = a
    return A, t_max


def save_fig(fig: plt.Figure, fig_dir: Path, stem: str):
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        out = fig_dir / f'{stem}.{ext}'
        fig.savefig(out)
        print(f'[plot] wrote {out}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — layer-wise probe accuracy
# ---------------------------------------------------------------------------

def plot_layer_probe_accuracy(by_lang: dict[str, list[dict]], fig_dir: Path):
    langs = list(by_lang.keys())
    n = len(langs)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.8), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, lang in zip(axes, langs):
        rows = by_lang[lang]
        layers = [as_int(r, 'layer') for r in rows]
        for key, (color, marker, label) in ACC_SERIES.items():
            ys = [as_float(r, key) for r in rows]
            if marker == '--':
                ax.plot(layers, ys, linestyle='--', color=color, label=label, linewidth=1.4)
            else:
                ax.plot(
                    layers, ys, color=color, marker=marker, markersize=4,
                    markevery=max(1, len(layers) // 12), label=label,
                )
        ax.set_title(f'target = {lang.upper()} (one-vs-rest)')
        ax.set_xlabel('Layer index')
        ax.set_ylim(0.0, 1.02)
        ax.yaxis.grid(True)
        ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)

    axes[0].set_ylabel('Language-probe accuracy')
    axes[-1].legend(loc='lower left', bbox_to_anchor=(1.02, 0.0))
    fig.suptitle(
        'INLP language probe across layers',
        fontsize=13, y=1.02,
    )
    fig.text(
        0.5, -0.02,
        'Source: inlp_runner summary · acc_first / acc_last / fresh probe after P⊥ · chance = majority baseline',
        ha='center', fontsize=8, color='#5D6D7E',
    )
    fig.tight_layout()
    save_fig(fig, fig_dir, '01_layer_probe_accuracy')


# ---------------------------------------------------------------------------
# Figure 2 — Δlang + removed directions
# ---------------------------------------------------------------------------

def plot_delta_and_removed(by_lang: dict[str, list[dict]], fig_dir: Path):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)

    for lang, rows in by_lang.items():
        layers = [as_int(r, 'layer') for r in rows]
        delta = [as_float(r, 'acc_first') - as_float(r, 'acc_after') for r in rows]
        removed = [as_int(r, 'n_removed') for r in rows]
        c = lang_color(lang)
        ax0.plot(layers, delta, color=c, marker='o', markersize=4,
                 markevery=max(1, len(layers) // 12), label=lang.upper())
        ax1.plot(layers, removed, color=c, marker='s', markersize=4,
                 markevery=max(1, len(layers) // 12), label=lang.upper())

    ax0.axhline(0.0, color='#BDC3C7', linewidth=1.0)
    ax0.set_ylabel(r'$\Delta_{\mathrm{lang}}$  (acc first − acc after)')
    ax0.set_title('Language information removed by INLP')
    ax0.yaxis.grid(True)
    ax0.legend(loc='best', title='target')

    ax1.set_xlabel('Layer index')
    ax1.set_ylabel('# removed directions  k')
    ax1.set_title('Size of language subspace identified by INLP')
    ax1.yaxis.grid(True)
    ax1.legend(loc='best', title='target')

    fig.text(
        0.5, -0.01,
        'Source: inlp_runner summary · one-vs-rest per target language · higher Δlang ⇒ more linearly readable language identity removed',
        ha='center', fontsize=8, color='#5D6D7E',
    )
    fig.tight_layout()
    save_fig(fig, fig_dir, '02_delta_lang_and_removed')


def _has_cap(rows: list[dict]) -> bool:
    if not rows or 'delta_cap' not in rows[0]:
        return False
    try:
        return not any(
            (as_float(r, 'delta_cap') != as_float(r, 'delta_cap'))  # NaN check
            for r in rows
        )
    except (KeyError, ValueError):
        return False


def plot_delta_cap(by_lang: dict[str, list[dict]], fig_dir: Path):
    """Paper-style Δcap / capability probe before vs after INLP (Fig. 3 blue)."""
    # skip if capability columns missing / all NaN
    if not any(_has_cap(rows) for rows in by_lang.values()):
        # still plot if at least some finite values
        any_finite = False
        for rows in by_lang.values():
            for r in rows:
                if 'delta_cap' in r:
                    try:
                        v = as_float(r, 'delta_cap')
                        if v == v:
                            any_finite = True
                    except ValueError:
                        pass
        if not any_finite:
            print('[plot] skip 05_delta_cap (no capability columns in summary)')
            return

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)

    for lang, rows in by_lang.items():
        layers = [as_int(r, 'layer') for r in rows]
        before = [as_float(r, 'cap_acc_before') for r in rows]
        after = [as_float(r, 'cap_acc_after') for r in rows]
        delta = [as_float(r, 'delta_cap') for r in rows]
        c = lang_color(lang)
        ax0.plot(layers, before, color=c, marker='o', markersize=3.5,
                 markevery=max(1, len(layers) // 12), linestyle='-',
                 label=f'{lang.upper()} before')
        ax0.plot(layers, after, color=c, marker='D', markersize=3.5,
                 markevery=max(1, len(layers) // 12), linestyle='--',
                 label=f'{lang.upper()} after')
        ax1.plot(layers, delta, color=c, marker='s', markersize=4,
                 markevery=max(1, len(layers) // 12), label=lang.upper())

    ax0.set_ylabel('Capability-probe accuracy')
    ax0.set_title('Reward probe before / after removing language directions')
    ax0.set_ylim(0.0, 1.02)
    ax0.yaxis.grid(True)
    ax0.legend(loc='best', ncol=2, fontsize=8)

    ax1.axhline(0.0, color='#BDC3C7', linewidth=1.0)
    ax1.set_xlabel('Layer index')
    ax1.set_ylabel(r'$\Delta_{\mathrm{cap}}$  (acc after − acc before)')
    ax1.set_title('Capability change after INLP (paper entanglement signal)')
    ax1.yaxis.grid(True)
    ax1.legend(loc='best', title='target')

    fig.text(
        0.5, -0.01,
        'Source: inlp_runner summary · linear BCE reward probe (trainer/scan_layers recipe) · '
        'Δcap<0 ⇒ language dirs facilitated capability; Δcap>0 ⇒ interference',
        ha='center', fontsize=8, color='#5D6D7E',
    )
    fig.tight_layout()
    save_fig(fig, fig_dir, '05_delta_cap')


# ---------------------------------------------------------------------------
# Figure 3 — iteration heatmap
# ---------------------------------------------------------------------------

def plot_iteration_heatmap(inlp_dir: Path, by_lang: dict[str, list[dict]], fig_dir: Path):
    langs = list(by_lang.keys())
    n = len(langs)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.2), sharey=True)
    if n == 1:
        axes = [axes]

    last_im = None
    for ax, lang in zip(axes, langs):
        rows = by_lang[lang]
        layers = [as_int(r, 'layer') for r in rows]
        A, t_max = load_accs_matrix(inlp_dir, lang, layers)
        if t_max == 0:
            ax.set_visible(False)
            continue
        last_im = ax.imshow(
            A,
            aspect='auto',
            origin='lower',
            cmap='YlOrBr',
            vmin=0.0,
            vmax=1.0,
            extent=(-0.5, t_max - 0.5, layers[0] - 0.5, layers[-1] + 0.5),
            interpolation='nearest',
        )
        ax.set_title(f'target = {lang.upper()}')
        ax.set_xlabel('INLP iteration')
        # y ticks at a sensible stride
        stride = max(1, len(layers) // 8)
        ax.set_yticks(layers[::stride])

    axes[0].set_ylabel('Layer index')
    fig.subplots_adjust(right=0.88 if n > 1 else 0.86, wspace=0.18)
    if last_im is not None:
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label('Classifier accuracy')
    fig.suptitle('INLP convergence trajectory (accuracy per iteration)', fontsize=13, y=0.98)
    fig.text(
        0.5, 0.02,
        'Source: target_*/accs_layer*.pt · blank cells = iteration not run (already at chance)',
        ha='center', fontsize=8, color='#5D6D7E',
    )
    save_fig(fig, fig_dir, '03_inlp_iteration_heatmap')


# ---------------------------------------------------------------------------
# Figure 4 — selected layer curves
# ---------------------------------------------------------------------------

def plot_iteration_curves(inlp_dir: Path, by_lang: dict[str, list[dict]], fig_dir: Path):
    langs = list(by_lang.keys())
    n = len(langs)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.8), sharey=True)
    if n == 1:
        axes = [axes]

    # pick up to 5 layers spanning depth
    for ax, lang in zip(axes, langs):
        rows = by_lang[lang]
        layers = [as_int(r, 'layer') for r in rows]
        if not layers:
            continue
        pick_idx = np.unique(np.linspace(0, len(layers) - 1, num=min(5, len(layers)), dtype=int))
        pick = [layers[i] for i in pick_idx]
        A, t_max = load_accs_matrix(inlp_dir, lang, pick)
        chance = as_float(rows[0], 'chance_acc')
        cmap = plt.get_cmap('cividis')
        for i, layer in enumerate(pick):
            ys = A[i]
            xs = np.arange(1, len(ys) + 1)
            mask = ~np.isnan(ys)
            ax.plot(
                xs[mask], ys[mask],
                color=cmap(0.15 + 0.7 * i / max(len(pick) - 1, 1)),
                marker='o', markersize=3.5,
                label=f'L{layer}',
            )
        ax.axhline(chance, color='#7F8C8D', linestyle='--', linewidth=1.2, label='chance')
        ax.set_title(f'target = {lang.upper()}')
        ax.set_xlabel('INLP iteration')
        ax.set_ylim(0.0, 1.02)
        ax.yaxis.grid(True)
        ax.legend(loc='best', ncol=2)

    axes[0].set_ylabel('Classifier accuracy')
    fig.suptitle('INLP accuracy drop over iterations (selected layers)', fontsize=13)
    fig.text(
        0.5, -0.02,
        'Source: target_*/accs_layer*.pt · dashed = majority-class chance',
        ha='center', fontsize=8, color='#5D6D7E',
    )
    fig.tight_layout()
    save_fig(fig, fig_dir, '04_inlp_iteration_curves')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def set_args():
    p = argparse.ArgumentParser(description='Plot multilingual INLP results.')
    p.add_argument(
        '--inlp_dir',
        type=str,
        required=True,
        help='directory produced by inlp_runner (contains summary_all.csv / target_*).',
    )
    p.add_argument(
        '--fig_dir',
        type=str,
        default='',
        help='where to write figures (default: <inlp_dir>/figs).',
    )
    return p.parse_args()


def main():
    args = set_args()
    apply_style()
    inlp_dir = Path(args.inlp_dir)
    fig_dir = Path(args.fig_dir) if args.fig_dir else inlp_dir / 'figs'

    rows = load_summary(inlp_dir)
    by_lang = group_by_lang(rows)
    print(f'[plot] langs={list(by_lang)}  rows={len(rows)}  -> {fig_dir}')

    plot_layer_probe_accuracy(by_lang, fig_dir)
    plot_delta_and_removed(by_lang, fig_dir)
    plot_iteration_heatmap(inlp_dir, by_lang, fig_dir)
    plot_iteration_curves(inlp_dir, by_lang, fig_dir)
    plot_delta_cap(by_lang, fig_dir)
    print('[plot] done.')


if __name__ == '__main__':
    main()

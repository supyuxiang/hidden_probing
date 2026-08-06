"""
5x5 cross-lingual eval: each language-tuned BERT on all 5 validation sets.
Uses inference.py helpers; saves acc matrix + heatmap.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import (  # noqa: E402
    BertDataset,
    build_model_from_config,
    collate_fn,
    evaluate,
    load_checkpoint,
    resolve_data_path,
)

LANGS = ["en", "zh", "es", "vi", "tr"]
CKPT_ROOT = Path("/root/autodl-tmp/exp2_bert/ckpt")
OUT_DIR = Path("/root/autodl-tmp/exp2_bert/eval")


def _fast_forward(self, texts: str | list[str]) -> torch.Tensor:
    """Same as Classifier_Bert.forward but without per-batch gc/empty_cache."""
    if isinstance(texts, str):
        texts = [texts]
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
    hs = o.hidden_states[self.layer_idx]
    pooled = self.pooling_fn(hs, attention_mask)
    return self.classifier_head(pooled)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    acc = np.zeros((len(LANGS), len(LANGS)), dtype=np.float64)
    details = {}

    for i, train_lang in enumerate(LANGS):
        ckpt_path = CKPT_ROOT / f"bert_tuned_with_{train_lang}" / "best.pt"
        assert ckpt_path.is_file(), ckpt_path
        print(f"\n=== train_lang={train_lang} ckpt={ckpt_path} ===")
        state_dict, cfg = load_checkpoint(ckpt_path, device)
        w = state_dict["classifier_head.weight"]
        num_classes = int(w.shape[0])

        model = build_model_from_config(cfg, num_classes=num_classes, device=device)
        model.forward = types.MethodType(_fast_forward, model)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        ds_cfg = cfg.get("dataset", {})
        text_field = ds_cfg.get("text_field", "sentence")
        label_field = ds_cfg.get("label_field", "label")
        batch_size = int(cfg.get("train", {}).get("batch_size", 32))

        for j, eval_lang in enumerate(LANGS):
            data_path = resolve_data_path(eval_lang, None)
            dataset = BertDataset(data_path, text_field=text_field, label_field=label_field)
            keep = [k for k, y in enumerate(dataset.labels) if y >= 0]
            if len(keep) != len(dataset):
                dataset.texts = [dataset.texts[k] for k in keep]
                dataset.labels = [dataset.labels[k] for k in keep]

            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
            )
            metrics = evaluate(model, loader, device)
            acc[i, j] = metrics["acc"]
            details[f"{train_lang}->{eval_lang}"] = {
                "acc": metrics["acc"],
                "correct": metrics["correct"],
                "n": metrics["n"],
                "data_path": str(data_path),
            }
            print(
                f"  {train_lang}->{eval_lang}: acc={metrics['acc']:.4f} "
                f"({metrics['correct']}/{metrics['n']})"
            )

        del model
        torch.cuda.empty_cache()

    # save matrix
    mat_path = OUT_DIR / "crosslingual_acc_5x5.json"
    with open(mat_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_langs": LANGS,
                "eval_langs": LANGS,
                "acc": acc.tolist(),
                "details": details,
            },
            f,
            indent=2,
        )
    print(f"\nsaved {mat_path}")

    # heatmap: rows = train lang, cols = eval lang
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    sns.heatmap(
        acc,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.5,
        vmax=1.0,
        xticklabels=LANGS,
        yticklabels=LANGS,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "accuracy"},
        ax=ax,
    )
    ax.set_xlabel("eval language")
    ax.set_ylabel("train language (ckpt)")
    ax.set_title("SST-2 cross-lingual transfer (BERT mBERT)")
    fig.tight_layout()
    fig_path = OUT_DIR / "crosslingual_acc_heatmap.png"
    fig.savefig(fig_path, dpi=200)
    print(f"saved {fig_path}")

    print("\nacc matrix (rows=train, cols=eval):")
    print("      " + "  ".join(f"{l:>6}" for l in LANGS))
    for i, tl in enumerate(LANGS):
        print(f"{tl:>4} " + "  ".join(f"{acc[i, j]:6.3f}" for j in range(len(LANGS))))


if __name__ == "__main__":
    main()

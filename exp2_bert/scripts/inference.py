"""
Eval a fine-tuned BERT classifier on a (possibly multilingual) JSON dataset.

Example:
  source /root/autodl-tmp/miniconda3/bin/activate bert1
  python /root/hidden_prob/exp2_bert/scripts/inference.py \\
      --ckpt_path /root/autodl-tmp/exp2_bert/ckpt/bert_tuned_with_en/best.pt \\
      --data_path /root/autodl-tmp/data/text_classification/glue_sst2/validation_zh.json

  # or resolve data path by language (SST-2 convention):
  python /root/hidden_prob/exp2_bert/scripts/inference.py \\
      --ckpt_path /root/autodl-tmp/exp2_bert/ckpt/bert_tuned_with_en/best.pt \\
      --language zh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from exp2_bert.trainer import BertDataset, Classifier_Bert, collate_fn  # noqa: E402

DEFAULT_DATA_DIR = Path("/root/autodl-tmp/data/text_classification/glue_sst2")


def resolve_data_path(language: str | None, data_path: str | None) -> Path:
    if data_path is not None:
        return Path(data_path)
    if language is None:
        raise ValueError("Provide --data_path or --language")
    language = language.lower()
    assert language in {"en", "zh", "es", "vi", "tr"}, f"unsupported language={language}"
    if language == "en":
        return DEFAULT_DATA_DIR / "validation.json"
    return DEFAULT_DATA_DIR / f"validation_{language}.json"


def load_checkpoint(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    assert "model_state_dict" in ckpt, f"missing model_state_dict in {ckpt_path}"
    cfg = ckpt.get("config", None)
    if cfg is None:
        raise ValueError(f"checkpoint has no config: {ckpt_path}")
    return ckpt["model_state_dict"], cfg


def build_model_from_config(cfg: dict, num_classes: int, device: torch.device) -> Classifier_Bert:
    mcfg = cfg["model"]
    model = Classifier_Bert(
        bert_path=mcfg["model_path"],
        output_dim=num_classes,
        layer_idx=int(mcfg.get("layer_idx", -1)),
        pooling_mode=str(mcfg.get("pooling_mode", "cls")),
        freeze_bert=True,  # eval only
        max_length=int(mcfg.get("max_length", 128)),
        device=device,
    )
    return model


@torch.no_grad()
def evaluate(
    model: Classifier_Bert,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    total_correct = 0
    total_n = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for texts, labels in tqdm(dataloader, desc="inference", leave=False):
        labels = labels.to(device)
        logits = model(texts)
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == labels).sum().item())
        total_n += labels.size(0)
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = total_correct / max(total_n, 1)
    return {
        "acc": acc,
        "n": total_n,
        "correct": total_correct,
        "preds": all_preds,
        "labels": all_labels,
    }


def parse_args():
    p = argparse.ArgumentParser(description="BERT classification inference / eval")
    p.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="path to best.pt / last_epoch.pt saved by trainer.py",
    )
    p.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="eval JSON list; if omitted, use --language with SST-2 defaults",
    )
    p.add_argument(
        "--language",
        type=str,
        default=None,
        choices=["en", "zh", "es", "vi", "tr"],
        help="shortcut for SST-2 validation / validation_{lang}.json",
    )
    p.add_argument("--text_field", type=str, default=None, help="override text field name")
    p.add_argument("--label_field", type=str, default=None, help="override label field name")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="override num_classes; default = infer from ckpt head weight",
    )
    p.add_argument(
        "--save_pred_path",
        type=str,
        default=None,
        help="optional path to dump preds/labels/acc as JSON",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("using cpu")

    ckpt_path = Path(args.ckpt_path)
    data_path = resolve_data_path(args.language, args.data_path)
    assert ckpt_path.is_file(), f"ckpt not found: {ckpt_path}"
    assert data_path.is_file(), f"data not found: {data_path}"

    state_dict, cfg = load_checkpoint(ckpt_path, device)

    ds_cfg = cfg.get("dataset", {})
    text_field = args.text_field or ds_cfg.get("text_field", "sentence")
    label_field = args.label_field or ds_cfg.get("label_field", "label")

    dataset = BertDataset(data_path, text_field=text_field, label_field=label_field)
    # drop hidden labels (e.g. SST-2 test label=-1) if any slipped in
    n_before = len(dataset)
    keep = [i for i, y in enumerate(dataset.labels) if y >= 0]
    if len(keep) != n_before:
        dataset.texts = [dataset.texts[i] for i in keep]
        dataset.labels = [dataset.labels[i] for i in keep]
        dataset.num_classes = len(set(dataset.labels))
        print(f"[data] dropped {n_before - len(keep)} hidden-label rows")
    if args.num_classes is not None:
        num_classes = args.num_classes
    else:
        # classifier_head.weight: (C, H)
        w = state_dict.get("classifier_head.weight")
        assert w is not None, "classifier_head.weight missing in state_dict"
        num_classes = int(w.shape[0])

    print(
        f"[data] path={data_path} n={len(dataset)} "
        f"text_field={text_field} label_field={label_field} "
        f"labels={sorted(set(dataset.labels))}"
    )
    print(f"[ckpt] {ckpt_path} num_classes={num_classes}")

    model = build_model_from_config(cfg, num_classes=num_classes, device=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    metrics = evaluate(model, loader, device)
    print(
        f"[result] language={args.language} acc={metrics['acc']:.4f} "
        f"correct={metrics['correct']}/{metrics['n']}"
    )

    if args.save_pred_path:
        out = {
            "ckpt_path": str(ckpt_path),
            "data_path": str(data_path),
            "language": args.language,
            "acc": metrics["acc"],
            "n": metrics["n"],
            "correct": metrics["correct"],
            "preds": metrics["preds"],
            "labels": metrics["labels"],
        }
        out_path = Path(args.save_pred_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()

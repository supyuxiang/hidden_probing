"""
Simplified SFT example on custom data using the vendored verl trainer.

Dataset format (JSON list):
  [
    {"question": "...", "golden_res": "..."},
    ...
  ]

Pipeline:
  1) Convert JSON -> parquet with `messages` (user/assistant turns) for MultiTurnSFTDataset
  2) Launch `torchrun -m verl.trainer.sft_trainer` with LoRA + bf16 FSDP

Example:
  source /root/autodl-tmp/miniconda3/bin/activate verl2
  export PYTHONPATH=/root/hidden_prob/exp3_fine_tuning/verl:$PYTHONPATH

  python /root/hidden_prob/exp3_fine_tuning/example_sft.py \
      --data_path /path/to/train.json \
      --val_data_path /path/to/val.json \
      --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
      --save_path /root/autodl-tmp/exp3_sft/qwen25_3b_lora \
      --nproc_per_node 1 \
      --lora_rank 64 \
      --micro_batch_size_per_gpu 4 \
      --max_length 1024 \
      --total_epochs 2
"""


import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

VERL_ROOT = Path(__file__).resolve().parent / "verl"
DEFAULT_MODEL = "/root/autodl-tmp/models/Qwen2.5-3B-Instruct"
DEFAULT_WORK_DIR = Path("/root/autodl-tmp/exp3_sft/data")


def load_qa_json(path: str | Path) -> list[dict]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and len(data) > 0, f"empty or invalid json: {path}"
    for i, row in enumerate(data):
        assert "question" in row and "golden_res" in row, (
            f"row {i} missing question/golden_res; keys={list(row.keys())}"
        )
    return data


def rows_to_messages(data: list[dict], system_prompt: str | None) -> list[dict]:
    out: list[dict] = []
    for row in data:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": str(row["question"])})
        messages.append({"role": "assistant", "content": str(row["golden_res"])})
        out.append({"messages": messages})
    return out


def save_parquet(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[data] wrote {len(rows)} samples -> {path}")
    return path


def prepare_parquet(
    data_path: str,
    val_data_path: str | None,
    work_dir: Path,
    val_ratio: float,
    seed: int,
    system_prompt: str | None,
) -> tuple[Path, Path | None]:
    train_raw = load_qa_json(data_path)
    if val_data_path:
        val_raw = load_qa_json(val_data_path)
    elif val_ratio > 0:
        import random

        rng = random.Random(seed)
        idx = list(range(len(train_raw)))
        rng.shuffle(idx)
        n_val = max(1, int(len(train_raw) * val_ratio)) if len(train_raw) > 1 else 0
        if n_val == 0:
            val_raw = []
        else:
            val_idx = set(idx[:n_val])
            val_raw = [train_raw[i] for i in range(len(train_raw)) if i in val_idx]
            train_raw = [train_raw[i] for i in range(len(train_raw)) if i not in val_idx]
    else:
        val_raw = []

    train_pq = save_parquet(rows_to_messages(train_raw, system_prompt), work_dir / "train.parquet")
    val_pq = None
    if val_raw:
        val_pq = save_parquet(rows_to_messages(val_raw, system_prompt), work_dir / "val.parquet")
    return train_pq, val_pq


def parse_args():
    p = argparse.ArgumentParser(description="Simplified verl SFT (Qwen2.5-3B + LoRA + bf16)")
    p.add_argument("--data_path", type=str, required=True, help="JSON list[{question,golden_res}]")
    p.add_argument("--val_data_path", type=str, default=None, help="optional val JSON; else split from train")
    p.add_argument("--val_ratio", type=float, default=0.05, help="used when --val_data_path is omitted")
    p.add_argument("--work_dir", type=str, default=str(DEFAULT_WORK_DIR), help="where to write parquet")
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--save_path", type=str, default="/root/autodl-tmp/exp3_sft/qwen25_3b_lora")
    p.add_argument("--nproc_per_node", type=int, default=1)
    p.add_argument("--micro_batch_size_per_gpu", type=int, default=4)
    p.add_argument("--train_batch_size", type=int, default=16, help="global batch size")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--total_epochs", type=int, default=2)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_targets", type=str, default="all-linear")
    p.add_argument("--sp_size", type=int, default=1, help="Ulysses sequence parallel size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful assistant.",
        help="set empty string to disable system turn",
    )
    p.add_argument("--project_name", type=str, default="exp3-sft")
    p.add_argument("--experiment_name", type=str, default="qwen25-3b-lora")
    p.add_argument(
        "--logger",
        type=str,
        default="console",
        help='comma-separated: console,wandb,swanlab  e.g. "console" or "console,wandb"',
    )
    p.add_argument("--dry_run", action="store_true", help="only convert data + print command")
    p.add_argument("extra", nargs="*", help="extra Hydra overrides forwarded to sft_trainer")
    return p.parse_args()


def build_cmd(args, train_pq: Path, val_pq: Path | None) -> list[str]:
    logger_list = [x.strip() for x in args.logger.split(",") if x.strip()]
    logger_hydra = "[" + ",".join(f"'{x}'" for x in logger_list) + "]"

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={args.nproc_per_node}",
        "-m",
        "verl.trainer.sft_trainer",
        f"data.train_files={train_pq}",
        f"data.messages_key=messages",
        f"data.micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.max_length={args.max_length}",
        "data.truncation=right",
        "data.ignore_input_ids_mismatch=True",
        "data.use_dynamic_bsz=False",
        f"optim.lr={args.lr}",
        "engine=fsdp",
        "engine.dtype=bfloat16",
        f"engine.ulysses_sequence_parallel_size={args.sp_size}",
        f"model.path={args.model_path}",
        "model.trust_remote_code=True",
        "model.use_remove_padding=True",
        "model.enable_gradient_checkpointing=True",
        f"model.lora_rank={args.lora_rank}",
        f"model.lora_alpha={args.lora_alpha}",
        f"model.target_modules={args.lora_targets}",
        f"trainer.default_local_dir={args.save_path}",
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={args.experiment_name}",
        f"trainer.logger={logger_hydra}",
        f"trainer.total_epochs={args.total_epochs}",
        f"trainer.n_gpus_per_node={args.nproc_per_node}",
        f"trainer.seed={args.seed}",
        "trainer.device=cuda",
    ]
    if val_pq is not None:
        cmd.append(f"data.val_files={val_pq}")
    else:
        cmd.append("data.val_files=null")
    cmd.extend(args.extra)
    return cmd


def main():
    args = parse_args()
    assert VERL_ROOT.is_dir(), f"missing vendored verl at {VERL_ROOT}"

    # Prefer local vendored verl
    env = os.environ.copy()
    py_path = str(VERL_ROOT)
    env["PYTHONPATH"] = py_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    system_prompt = args.system_prompt.strip() or None
    train_pq, val_pq = prepare_parquet(
        data_path=args.data_path,
        val_data_path=args.val_data_path,
        work_dir=Path(args.work_dir),
        val_ratio=args.val_ratio,
        seed=args.seed,
        system_prompt=system_prompt,
    )

    cmd = build_cmd(args, train_pq, val_pq)
    print("[cmd]", " ".join(cmd))
    if args.dry_run:
        print("[dry_run] skip launching trainer")
        return

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, env=env)
    print(f"[done] checkpoints under {args.save_path}")


if __name__ == "__main__":
    main()

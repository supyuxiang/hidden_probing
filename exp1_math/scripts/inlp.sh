#!/usr/bin/env bash
# Multilingual INLP on pooled train hiddens (one-vs-rest per target language).
# Phase 1: en + zh only (available hs). Extend --langs when es/vi/tr train hs exist.
# After language INLP, runs capability (reward) probes before/after projection (Δcap).
#
# Templates use fixed train split (no {split}): hs_math_train_{lang}_... / reward_math_train_{lang}_...

set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HS_DIR="/root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct"
REWARD_DIR="/root/autodl-tmp/exp1_math/judge"
OUT_DIR="/root/autodl-tmp/exp1_math/inlp/Qwen2.5-3B-Instruct/en_zh_train"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python /root/hidden_prob/exp1_math/inlp_runner.py \
    --hs_dir "${HS_DIR}" \
    --reward_dir "${REWARD_DIR}" \
    --langs en,zh \
    --hs_template 'hs_math_train_{lang}_n8_tokens1024.pt' \
    --reward_template 'reward_math_train_{lang}_n8_t1.5_tokens1024.pt' \
    --target_lang all \
    --layer_indices all \
    --T 15 \
    --epochs_per_iter 30 \
    --cap_epochs 50 \
    --cap_scope target \
    --batch_size 512 \
    --lr 1e-3 \
    --weight_decay 0.0 \
    --chance_tolerance 0.02 \
    --split_ratio 0.95 \
    --seed 42 \
    --out_dir "${OUT_DIR}"

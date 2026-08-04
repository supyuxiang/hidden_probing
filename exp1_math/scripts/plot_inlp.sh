#!/usr/bin/env bash
# Plot INLP results (run after scripts/inlp.sh finishes).

set -euo pipefail

INLP_DIR="${1:-/root/autodl-tmp/exp1_math/inlp/Qwen2.5-3B-Instruct/en_zh_train}"
FIG_DIR="${2:-${INLP_DIR}/figs}"

python /root/hidden_prob/exp1_math/plot_inlp.py \
    --inlp_dir "${INLP_DIR}" \
    --fig_dir "${FIG_DIR}"

echo "figures -> ${FIG_DIR}"

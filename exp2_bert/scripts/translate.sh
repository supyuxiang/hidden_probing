#!/usr/bin/env bash
# Translate SST-2 train/test into zh,es,vi,tr with Qwen2.5-32B-Instruct.
set -euo pipefail

PYTHON="${PYTHON:-/root/autodl-tmp/miniconda3/envs/verl3/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON}" /root/hidden_prob/exp2_bert/translate.py \
    --model_path /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
    --data_dir /root/autodl-tmp/data/text_classification/glue_sst2 \
    --save_dir /root/autodl-tmp/data/text_classification/glue_sst2 \
    --splits train,test \
    --langs zh,es,vi,tr \
    --max_tokens 256 \
    --temperature 0.0 \
    "$@"


# source /root/autodl-tmp/miniconda3/bin/activate verl3
# bash /root/hidden_prob/exp2_bert/scripts/translate.sh
# # 冒烟：
# CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp2_bert/translate.py --splits train --langs zh --limit 8

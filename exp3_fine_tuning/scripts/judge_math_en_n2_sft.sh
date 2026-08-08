#!/usr/bin/env bash
# Judge teacher SFT responses (golden_res) for exp3.

set -euo pipefail

judge_model=/root/autodl-tmp/models/Qwen2.5-32B-Instruct
gpu_util=0.95

data_path=/root/hidden_prob/exp3_fine_tuning/teacher/math_en_n2_sft.json
save_dir=/root/autodl-tmp/exp3_sft/judge
save_path="$save_dir"/reward_math_en_n2_sft.pt

mkdir -p "$save_dir"

CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/judge.py \
    --judge_model "$judge_model" \
    --data_path "$data_path" \
    --save_path "$save_path" \
    --gpu_memory_utilization "$gpu_util"

echo "done -> $save_path"

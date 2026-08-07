#!/usr/bin/env bash
# Multilingual INLP on pooled train3000 hiddens (one-vs-rest per target language).
# After language INLP, runs capability (reward) probes before/after projection (Δcap).

set -euo pipefail

models=(/root/autodl-tmp/models/Qwen2.5-3B-Instruct \
        /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
        /root/autodl-tmp/models/Qwen3-8B \
        /root/autodl-tmp/models/Llama3.2-8B-Instruct)

langs=en,zh,es,vi,tr

for model in "${models[@]}"; do
    model_name=$(basename "$model")
    tokens=1024
    if [ "$model" = "/root/autodl-tmp/models/Qwen3-8B" ]; then
        tokens=4096
    fi

    hs_dir=/root/autodl-tmp/exp1_math/hs/"$model_name"
    reward_dir=/root/autodl-tmp/exp1_math/judge/"$model_name"
    out_dir=/root/autodl-tmp/exp1_math/inlp/"$model_name"/train3000

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python /root/hidden_prob/exp1_math/inlp_runner.py \
        --hs_dir "$hs_dir" \
        --reward_dir "$reward_dir" \
        --langs "$langs" \
        --hs_template "hs_math_train3000_{lang}_n8_tokens${tokens}.pt" \
        --reward_template "reward_math_train3000_{lang}_n8_t1.5_tokens${tokens}.pt" \
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
        --out_dir "$out_dir"
done

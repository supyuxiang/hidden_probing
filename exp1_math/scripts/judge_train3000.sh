#!/usr/bin/env bash
# Judge sampled responses on the train3000 subsets.

judge_model=/root/autodl-tmp/models/Qwen2.5-32B-Instruct
gpu_util=0.95

models=(/root/autodl-tmp/models/Qwen2.5-3B-Instruct \
        /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
        /root/autodl-tmp/models/Qwen3-8B \
        /root/autodl-tmp/models/Llama3.2-8B-Instruct)

for model in "${models[@]}"; do
    model_name=$(basename "$model")
    data_dir=/root/hidden_prob/exp1_math/sampled/"$model_name"
    save_dir=/root/autodl-tmp/exp1_math/judge/"$model_name"
    tokens=1024
    if [ "$model" = "/root/autodl-tmp/models/Qwen3-8B" ]; then
        tokens=4096
    fi
    mkdir -p "$save_dir"
    for language in en zh es vi tr; do
        data_path="$data_dir"/res_math_train3000_${language}_n8_t1.5_tokens${tokens}.json
        save_path="$save_dir"/reward_math_train3000_${language}_n8_t1.5_tokens${tokens}.pt
        CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/judge.py \
            --judge_model "$judge_model" \
            --data_path "$data_path" \
            --save_path "$save_path" \
            --gpu_memory_utilization "$gpu_util"
    done
done

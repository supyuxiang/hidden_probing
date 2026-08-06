#!/usr/bin/env bash
# Sample responses on the fixed train_split3000 subsets (3000 questions / lang).

models=(/root/autodl-tmp/models/Qwen2.5-3B-Instruct \
        /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
        /root/autodl-tmp/models/Qwen3-8B \
        /root/autodl-tmp/models/Llama3.2-8B-Instruct)

data_dir=/root/hidden_prob/data/math

for model in "${models[@]}"; do
    model_name=$(basename "$model")
    save_dir=/root/hidden_prob/exp1_math/sampled/"$model_name"
    mkdir -p "$save_dir"
    for language in en zh es vi tr; do
        if [ "$language" = "en" ]; then
            data_path="${data_dir}/train_split3000.json"
        else
            data_path="${data_dir}/train_${language}_split3000.json"
        fi
        CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_res.py \
            --model_path "$model" \
            --data_path "$data_path" \
            --save_path "$save_dir"/res_math_train3000_${language}_n8_t1.5_tokens1024.json \
            --language_type ${language} \
            --max_tokens 1024 \
            --temperature 1.5 \
            --top_p 0.95 \
            --top_k 50 \
            --n 8
    done
done

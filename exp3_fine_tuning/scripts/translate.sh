#!/usr/bin/env bash
# Translate EN teacher SFT (golden_res) into zh / es / vi / tr.

set -euo pipefail

model_path=/root/autodl-tmp/models/Qwen2.5-32B-Instruct
data_path=/root/hidden_prob/exp3_fine_tuning/teacher/math_en_n2_sft.json
save_dir=/root/hidden_prob/exp3_fine_tuning/teacher
gpu_util=0.95
max_tokens=8192
max_model_len=16384
temperature=0.1

mkdir -p "$save_dir"

for tgt_lang in zh es vi tr; do
    save_path="$save_dir"/math_${tgt_lang}_n2_sft_translated.json
    echo "===== translate en -> ${tgt_lang} | save=${save_path} ====="
    CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp3_fine_tuning/translate_sft_data.py \
        --model_path "$model_path" \
        --data_path "$data_path" \
        --save_path "$save_path" \
        --src_lang en \
        --tgt_lang "$tgt_lang" \
        --temperature "$temperature" \
        --max_tokens "$max_tokens" \
        --max_model_len "$max_model_len" \
        --gpu_memory_utilization "$gpu_util"
done

echo "all done."

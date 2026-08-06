

models=(/root/autodl-tmp/models/Qwen2.5-3B-Instruct \
        /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
        /root/autodl-tmp/models/Qwen3-8B \
        /root/autodl-tmp/models/Llama3.2-8B-Instruct)

for model in "${models[@]}"; do
    model_name=$(basename "$model")
    save_dir=/root/hidden_prob/exp1_math/sampled/"$model_name"
    mkdir -p "$save_dir"
    for language in en zh es vi tr; do
        CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_res.py \
            --model_path "$model" \
            --data_path /root/hidden_prob/data/math/test.json \
            --save_path "$save_dir"/res_math_test_${language}_n8_t1.5_tokens1024.json \
            --language_type ${language} \
            --max_tokens 1024 \
            --temperature 1.5 \
            --top_p 0.95 \
            --top_k 50 \
            --n 8
    done
done



# #################      en         ####################
# # CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_res.py \
# #     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
# #     --data_path /root/hidden_prob/data/math/train.json \
# #     --save_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_en_n8_t1.5_tokens1024.json \
# #     --language_type en \
# #     --max_tokens 1024 \
# #     --temperature 1.5 \
# #     --top_p 0.95 \
# #     --top_k 50 \
# #     --n 8

# CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_res.py \
#     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
#     --data_path /root/hidden_prob/data/math/test.json \
#     --save_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_en_n8_t1.5_tokens1024.json \
#     --language_type en \
#     --max_tokens 1024 \
#     --temperature 1.5 \
#     --top_p 0.95 \
#     --top_k 50 \
#     --n 8


# #####################    zh     #####################
# # CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_res.py \
# #     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
# #     --data_path /root/hidden_prob/data/math/train.json \
# #     --save_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_zh_n8_t1.5_tokens1024.json \
# #     --language_type zh \
# #     --max_tokens 1024 \
# #     --temperature 1.5 \
# #     --top_p 0.95 \
# #     --top_k 50 \
# #     --n 8

# CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_res.py \
#     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
#     --data_path /root/hidden_prob/data/math/test.json \
#     --save_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_zh_n8_t1.5_tokens1024.json \
#     --language_type zh \
#     --max_tokens 1024 \
#     --temperature 1.5 \
#     --top_p 0.95 \
#     --top_k 50 \
#     --n 8

# #####################    es     #####################
# # CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_res.py \
# #     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
# #     --data_path /root/hidden_prob/data/math/train.json \
# #     --save_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_es_n8_t1.5_tokens1024.json \
# #     --language_type es \
# #     --max_tokens 1024 \
# #     --temperature 1.5 \
# #     --top_p 0.95 \
# #     --top_k 50 \
# #     --n 8

# CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_res.py \
#     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
#     --data_path /root/hidden_prob/data/math/test.json \
#     --save_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_es_n8_t1.5_tokens1024.json \
#     --language_type es \
#     --max_tokens 1024 \
#     --temperature 1.5 \
#     --top_p 0.95 \
#     --top_k 50 \
#     --n 8

# #####################    vi     #####################
# # CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_res.py \
# #     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
# #     --data_path /root/hidden_prob/data/math/train.json \
# #     --save_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_vi_n8_t1.5_tokens1024.json \
# #     --language_type vi \
# #     --max_tokens 1024 \
# #     --temperature 1.5 \
# #     --top_p 0.95 \
# #     --top_k 50 \
# #     --n 8

# CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_res.py \
#     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
#     --data_path /root/hidden_prob/data/math/test.json \
#     --save_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_vi_n8_t1.5_tokens1024.json \
#     --language_type vi \
#     --max_tokens 1024 \
#     --temperature 1.5 \
#     --top_p 0.95 \
#     --top_k 50 \
#     --n 8

# #####################    tr     #####################
# # CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_res.py \
# #     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
# #     --data_path /root/hidden_prob/data/math/train.json \
# #     --save_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_tr_n8_t1.5_tokens1024.json \
# #     --language_type tr \
# #     --max_tokens 1024 \
# #     --temperature 1.5 \
# #     --top_p 0.95 \
# #     --top_k 50 \
# #     --n 8

# CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_res.py \
#     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
#     --data_path /root/hidden_prob/data/math/test.json \
#     --save_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_tr_n8_t1.5_tokens1024.json \
#     --language_type tr \
#     --max_tokens 1024 \
#     --temperature 1.5 \
#     --top_p 0.95 \
#     --top_k 50 \
#     --n 8



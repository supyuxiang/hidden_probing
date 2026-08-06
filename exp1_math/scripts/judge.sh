

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
    mkdir -p "$save_dir"
    for language in en zh es vi tr; do
        data_path="$data_dir"/res_math_test_${language}_n8_t1.5_tokens1024.json
        save_path="$save_dir"/reward_math_test_${language}_n8_t1.5_tokens1024.pt
        CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/judge.py \
            --judge_model "$judge_model" \
            --data_path "$data_path" \
            --save_path "$save_path" \
            --gpu_memory_utilization "$gpu_util"
    done
done




# #####################    en     #####################
# # CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/judge.py \
# #     --judge_model /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
# #     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_en_n8_t1.5_tokens1024.json \
# #     --save_path /root/autodl-tmp/exp1_math/judge/reward_math_train_en_n8_t1.5_tokens1024.pt

# CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/judge.py \
#     --judge_model /root/autodl-tmp/models/Qwen2.5-32B-Instruct \
#     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-14B-Instruct/res_math_test_en_n8_t1.5_tokens1024.json \
#     --save_path /root/autodl-tmp/exp1_math/judge/Qwen2.5-14B-Instruct/reward_math_test_en_n8_t1.5_tokens1024.pt \
#     --gpu_memory_utilization 0.9

# #####################    zh     #####################
# # CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/judge.py \
# #     --judge_model /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
# #     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_zh_n8_t1.5_tokens1024.json \
# #     --save_path /root/autodl-tmp/exp1_math/judge/reward_math_train_zh_n8_t1.5_tokens1024.pt

# CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/judge.py \
#     --judge_model /root/autodl-tmp/models/Qwen2.5-32B-Instruct \
#     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-14B-Instruct/res_math_test_zh_n8_t1.5_tokens1024.json \
#     --save_path /root/autodl-tmp/exp1_math/judge/Qwen2.5-14B-Instruct/reward_math_test_zh_n8_t1.5_tokens1024.pt \
#     --gpu_memory_utilization 0.9

# #####################    es     #####################
# # CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/judge.py \
# #     --judge_model /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
# #     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_es_n8_t1.5_tokens1024.json \
# #     --save_path /root/autodl-tmp/exp1_math/judge/reward_math_train_es_n8_t1.5_tokens1024.pt

# CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/judge.py \
#     --judge_model /root/autodl-tmp/models/Qwen2.5-32B-Instruct \
#     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-14B-Instruct/res_math_test_es_n8_t1.5_tokens1024.json \
#     --save_path /root/autodl-tmp/exp1_math/judge/Qwen2.5-14B-Instruct/reward_math_test_es_n8_t1.5_tokens1024.pt \
#     --gpu_memory_utilization 0.9

# #####################    vi     #####################
# # CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/judge.py \
# #     --judge_model /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
# #     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_vi_n8_t1.5_tokens1024.json \
# #     --save_path /root/autodl-tmp/exp1_math/judge/reward_math_train_vi_n8_t1.5_tokens1024.pt

# CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/judge.py \
#     --judge_model /root/autodl-tmp/models/Qwen2.5-32B-Instruct \
#     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-14B-Instruct/res_math_test_vi_n8_t1.5_tokens1024.json \
#     --save_path /root/autodl-tmp/exp1_math/judge/Qwen2.5-14B-Instruct/reward_math_test_vi_n8_t1.5_tokens1024.pt \
#     --gpu_memory_utilization 0.9

# #####################    tr     #####################
# # CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/judge.py \
# #     --judge_model /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
# #     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_tr_n8_t1.5_tokens1024.json \
# #     --save_path /root/autodl-tmp/exp1_math/judge/reward_math_train_tr_n8_t1.5_tokens1024.pt

# CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/judge.py \
#     --judge_model /root/autodl-tmp/models/Qwen2.5-32B-Instruct \
#     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-14B-Instruct/res_math_test_tr_n8_t1.5_tokens1024.json \
#     --save_path /root/autodl-tmp/exp1_math/judge/Qwen2.5-14B-Instruct/reward_math_test_tr_n8_t1.5_tokens1024.pt \
#     --gpu_memory_utilization 0.9


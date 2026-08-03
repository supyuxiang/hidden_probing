

CUDA_VISIBLE_DEVICES=0,1 python /root/hidden_prob/exp1_math/translate.py \
    --model_path /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
    --data_path /root/hidden_prob/data/math/train.json \
    --target_language zh \
    --max_tokens 4096 \
    --top_p 0.95 \
    --top_k 50 \
    --temperature 0.0 \
    --n 1 \
    --save_path /root/hidden_prob/data/math/train_zh.json \
    --limit -1

CUDA_VISIBLE_DEVICES=0,1 python /root/hidden_prob/exp1_math/translate.py \
    --model_path /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
    --data_path /root/hidden_prob/data/math/test.json \
    --target_language zh \
    --max_tokens 4096 \
    --top_p 0.95 \
    --top_k 50 \
    --temperature 0.0 \
    --n 1 \
    --save_path /root/hidden_prob/data/math/test_zh.json \
    --limit -1

# #########################
# CUDA_VISIBLE_DEVICES=0,1 python /root/hidden_prob/exp1_math/translate.py \
#     --model_path /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
#     --data_path /root/hidden_prob/data/math/train.json \
#     --target_language es \
#     --max_tokens 4096 \
#     --top_p 0.95 \
#     --top_k 50 \
#     --temperature 0.0 \
#     --n 1 \
#     --save_path /root/hidden_prob/data/math/train_es.json \
#     --limit -1

# CUDA_VISIBLE_DEVICES=0,1 python /root/hidden_prob/exp1_math/translate.py \
#     --model_path /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
#     --data_path /root/hidden_prob/data/math/test.json \
#     --target_language es \
#     --max_tokens 4096 \
#     --top_p 0.95 \
#     --top_k 50 \
#     --temperature 0.0 \
#     --n 1 \
#     --save_path /root/hidden_prob/data/math/test_es.json \
#     --limit -1

# #########################
# CUDA_VISIBLE_DEVICES=0,1 python /root/hidden_prob/exp1_math/translate.py \
#     --model_path /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
#     --data_path /root/hidden_prob/data/math/train.json \
#     --target_language tr \
#     --max_tokens 4096 \
#     --top_p 0.95 \
#     --top_k 50 \
#     --temperature 0.0 \
#     --n 1 \
#     --save_path /root/hidden_prob/data/math/train_tr.json \
#     --limit -1

# CUDA_VISIBLE_DEVICES=0,1 python /root/hidden_prob/exp1_math/translate.py \
#     --model_path /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
#     --data_path /root/hidden_prob/data/math/test.json \
#     --target_language tr \
#     --max_tokens 4096 \
#     --top_p 0.95 \
#     --top_k 50 \
#     --temperature 0.0 \
#     --n 1 \
#     --save_path /root/hidden_prob/data/math/test_tr.json \
#     --limit -1

# #########################
# CUDA_VISIBLE_DEVICES=0,1 python /root/hidden_prob/exp1_math/translate.py \
#     --model_path /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
#     --data_path /root/hidden_prob/data/math/train.json \
#     --target_language vi \
#     --max_tokens 4096 \
#     --top_p 0.95 \
#     --top_k 50 \
#     --temperature 0.0 \
#     --n 1 \
#     --save_path /root/hidden_prob/data/math/train_vi.json \
#     --limit -1

# CUDA_VISIBLE_DEVICES=0,1 python /root/hidden_prob/exp1_math/translate.py \
#     --model_path /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
#     --data_path /root/hidden_prob/data/math/test.json \
#     --target_language vi \
#     --max_tokens 4096 \
#     --top_p 0.95 \
#     --top_k 50 \
#     --temperature 0.0 \
#     --n 1 \
#     --save_path /root/hidden_prob/data/math/test_vi.json \
#     --limit -1
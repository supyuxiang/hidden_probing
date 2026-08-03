

CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_res.py \
    --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
    --data_path /root/hidden_prob/data/math/train.json \
    --save_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_en_n8_t1.5_tokens1024.json \
    --language_type en \
    --max_tokens 1024 \
    --temperature 1.5 \
    --top_p 0.95 \
    --top_k 50 \
    --n 8

CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_res.py \
    --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
    --data_path /root/hidden_prob/data/math/test.json \
    --save_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_en_n8_t1.5_tokens1024.json \
    --language_type en \
    --max_tokens 1024 \
    --temperature 1.5 \
    --top_p 0.95 \
    --top_k 50 \
    --n 8

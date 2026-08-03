

CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/judge.py \
    --judge_model /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
    --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_en_n8_t1.5_tokens1024.json \
    --save_path /root/autodl-tmp/exp1_math/judge/reward_math_train_en_n8_t1.5_tokens1024.pt

CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/judge.py \
    --judge_model /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
    --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_en_n8_t1.5_tokens1024.json \
    --save_path /root/autodl-tmp/exp1_math/judge/reward_math_test_en_n8_t1.5_tokens1024.pt

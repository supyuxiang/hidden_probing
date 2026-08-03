# so slowly for sample all layers, do not used 

save_root=/root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct

for layer_idx in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35;do
    CUDA_VISIBLE_DEVICES=1 python /root/hidden_prob/exp1_math/sample_hidden.py \
        --batch_size 4 \
        --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
        --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_en_n32_t1.5_tokens1024.json \
        --save_path ${save_root}/hs_math_train_en_n32_tokens1024_layer${layer_idx}.pt \
        --language_type en \
        --layer_indices ${layer_idx} \
        --pooling_mode last 

    CUDA_VISIBLE_DEVICES=1 python /root/hidden_prob/exp1_math/sample_hidden.py \
        --batch_size 4 \
        --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
        --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_en_n32_t1.5_tokens1024.json \
        --save_path ${save_root}/hs_math_test_en_n32_tokens1024_layer${layer_idx}.pt \
        --language_type en \
        --layer_indices ${layer_idx} \
        --pooling_mode last 
done

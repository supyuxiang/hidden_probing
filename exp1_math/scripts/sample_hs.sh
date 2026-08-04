# use 

# #####################    en     #####################
# CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
#     --batch_size 4 \
#     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
#     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_en_n8_t1.5_tokens1024.json \
#     --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_train_en_n8_tokens1024.pt \
#     --language_type en \
#     --layer_indices all \
#     --pooling_mode last 

# CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
#     --batch_size 4 \
#     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
#     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_en_n8_t1.5_tokens1024.json \
#     --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_test_en_n8_tokens1024.pt \
#     --language_type en \
#     --layer_indices all \
#     --pooling_mode last 


########         test     ########
#####################    zh     #####################

CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
    --batch_size 20 \
    --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
    --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_zh_n8_t1.5_tokens1024.json \
    --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_test_zh_n8_tokens1024.pt \
    --language_type zh \
    --layer_indices all \
    --pooling_mode last 

#####################    es     #####################
CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
    --batch_size 20 \
    --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
    --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_es_n8_t1.5_tokens1024.json \
    --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_test_es_n8_tokens1024.pt \
    --language_type es \
    --layer_indices all \
    --pooling_mode last 

#####################    vi     #####################

CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \ 
    --batch_size 20 \
    --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
    --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_vi_n8_t1.5_tokens1024.json \
    --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_test_vi_n8_tokens1024.pt \
    --language_type vi \
    --layer_indices all \
    --pooling_mode last

########         tr         ###########

CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
    --batch_size 20 \
    --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
    --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_tr_n8_t1.5_tokens1024.json \
    --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_test_tr_n8_tokens1024.pt \
    --language_type tr \
    --layer_indices all \
    --pooling_mode last










########         train     ########

#####################    zh     #####################

CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
    --batch_size 20 \
    --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
    --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_zh_n8_t1.5_tokens1024.json \
    --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_train_zh_n8_tokens1024.pt \
    --language_type zh \
    --layer_indices all \
    --pooling_mode last 

#####################    es     #####################
CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
    --batch_size 4 \
    --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
    --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_es_n8_t1.5_tokens1024.json \
    --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_train_es_n8_tokens1024.pt \
    --language_type es \
    --layer_indices all \
    --pooling_mode last

#####################    vi     #####################

CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
    --batch_size 4 \
    --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
    --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_vi_n8_t1.5_tokens1024.json \
    --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_train_vi_n8_tokens1024.pt \
    --language_type vi \
    --layer_indices all \
    --pooling_mode last


#####################    tr     #####################
CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
    --batch_size 4 \
    --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
    --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_tr_n8_t1.5_tokens1024.json \
    --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_train_tr_n8_tokens1024.pt \
    --language_type tr \
    --layer_indices all \
    --pooling_mode last








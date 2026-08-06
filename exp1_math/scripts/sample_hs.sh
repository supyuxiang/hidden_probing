# # use 

models=(/root/autodl-tmp/models/Qwen2.5-3B-Instruct \
        /root/autodl-tmp/models/Qwen3-8B \
		/root/autodl-tmp/models/Qwen2.5-14B-Instruct \
		/root/autodl-tmp/models/Llama3.2-8B-Instruct)

languages=(en zh es vi tr)

batch_size=4
layer_indices=all
pooling_mode=last

for model in "${models[@]}"; do
	model_name=$(basename "$model")
	save_dir=/root/autodl-tmp/exp1_math/hs/"$model_name"
	mkdir -p "$save_dir"
	data_base=/root/hidden_prob/exp1_math/sampled/"$model_name"
	for language in "${languages[@]}"; do
		data_path="$data_base"/res_math_test_${language}_n8_t1.5_tokens1024.json
		save_path="$save_dir"/hs_math_test_${language}_n8_tokens1024.pt
		CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
			--batch_size $batch_size \
			--model_path "$model" \
			--data_path "$data_path" \
			--save_path "$save_path" \
			--language_type $language \
			--layer_indices $layer_indices \
			--pooling_mode $pooling_mode
	done
done



# ########         test     ########

# #####################    en     #####################

# CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
#     --batch_size 8 \
#     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
#     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_en_n8_t1.5_tokens1024.json \
#     --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_test_en_n8_tokens1024.pt \
#     --language_type en \
#     --layer_indices all \
#     --pooling_mode last 

# #####################    zh     #####################

# CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
#     --batch_size 8 \
#     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
#     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_zh_n8_t1.5_tokens1024.json \
#     --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_test_zh_n8_tokens1024.pt \
#     --language_type zh \
#     --layer_indices all \
#     --pooling_mode last 

# #####################    es     #####################
# CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
#     --batch_size 8 \
#     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
#     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_es_n8_t1.5_tokens1024.json \
#     --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_test_es_n8_tokens1024.pt \
#     --language_type es \
#     --layer_indices all \
#     --pooling_mode last 

# #####################    vi     #####################

# CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
#     --batch_size 8 \
#     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
#     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_vi_n8_t1.5_tokens1024.json \
#     --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_test_vi_n8_tokens1024.pt \
#     --language_type vi \
#     --layer_indices all \
#     --pooling_mode last

# #####################    tr     #####################

# CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
#     --batch_size 8 \
#     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
#     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_test_tr_n8_t1.5_tokens1024.json \
#     --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_test_tr_n8_tokens1024.pt \
#     --language_type tr \
#     --layer_indices all \
#     --pooling_mode last










# # ########         train     ########


# # #####################    en     #####################
# # CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
# #     --batch_size 4 \
# #     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
# #     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_en_n8_t1.5_tokens1024.json \
# #     --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_train_en_n8_tokens1024.pt \
# #     --language_type en \
# #     --layer_indices all \
# #     --pooling_mode last 


# # #####################    zh     #####################

# # CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
# #     --batch_size 20 \
# #     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
# #     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_zh_n8_t1.5_tokens1024.json \
# #     --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_train_zh_n8_tokens1024.pt \
# #     --language_type zh \
# #     --layer_indices all \
# #     --pooling_mode last 

# # #####################    es     #####################
# # CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
# #     --batch_size 4 \
# #     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
# #     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_es_n8_t1.5_tokens1024.json \
# #     --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_train_es_n8_tokens1024.pt \
# #     --language_type es \
# #     --layer_indices all \
# #     --pooling_mode last

# # #####################    vi     #####################

# # CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
# #     --batch_size 4 \
# #     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
# #     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_vi_n8_t1.5_tokens1024.json \
# #     --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_train_vi_n8_tokens1024.pt \
# #     --language_type vi \
# #     --layer_indices all \
# #     --pooling_mode last


# # #####################    tr     #####################
# # CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/exp1_math/sample_hidden.py \
# #     --batch_size 4 \
# #     --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
# #     --data_path /root/hidden_prob/exp1_math/sampled/Qwen2.5-3B-Instruct/res_math_train_tr_n8_t1.5_tokens1024.json \
# #     --save_path /root/autodl-tmp/exp1_math/hs/Qwen2.5-3B-Instruct/hs_math_train_tr_n8_tokens1024.pt \
# #     --language_type tr \
# #     --layer_indices all \
# #     --pooling_mode last








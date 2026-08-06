#!/usr/bin/env bash
# Sample hidden states on the train3000 sampled responses.

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
		data_path="$data_base"/res_math_train3000_${language}_n8_t1.5_tokens1024.json
		save_path="$save_dir"/hs_math_train3000_${language}_n8_tokens1024.pt
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

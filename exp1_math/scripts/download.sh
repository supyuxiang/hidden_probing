export HF_HUB_DISABLE_XET=1
source /etc/network_turbo
mkdir -p /root/autodl-tmp/models/Qwen3-8B
hf download Qwen/Qwen3-8B \
  --local-dir /root/autodl-tmp/models/Qwen3-8B \
  --max-workers 8


export HF_HUB_DISABLE_XET=1
source /etc/network_turbo
mkdir -p /root/autodl-tmp/models/Qwen2.5-3B-Instruct
hf download Qwen/Qwen2.5-3B-Instruct \
    --local-dir /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
    --max-workers 8


export HF_HUB_DISABLE_XET=1
source /etc/network_turbo
mkdir -p /root/autodl-tmp/models/Qwen2.5-14B-Instruct/
hf download Qwen/Qwen2.5-14B-Instruct \
    --local-dir /root/autodl-tmp/models/Qwen2.5-14B-Instruct \
    --max-workers 8


export HF_HUB_DISABLE_XET=1
source /etc/network_turbo
mkdir -p /root/autodl-tmp/models/bert-base-multilingual-cased/
hf download google-bert/bert-base-multilingual-cased \
    --local-dir /root/autodl-tmp/models/bert-base-multilingual-cased \
    --max-workers 8
 

export HF_HUB_DISABLE_XET=1
source /etc/network_turbo
mkdir -p /root/autodl-tmp/models/Qwen2.5-32B-Instruct/
hf download Qwen/Qwen2.5-32B-Instruct \
    --local-dir /root/autodl-tmp/models/Qwen2.5-32B-Instruct \
    --max-workers 8



# voidful/Llama-3.2-8B-Instruct
export HF_TOKEN="Your HF_TOKEN"
export HF_HUB_DISABLE_XET=1
source /etc/network_turbo
mkdir -p /root/autodl-tmp/models/Llama3.2-8B-Instruct
hf download voidful/Llama-3.2-8B-Instruct \
    --local-dir /root/autodl-tmp/models/Llama3.2-8B-Instruct \
    --max-workers 8




# deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
export HF_HUB_DISABLE_XET=1
source /etc/network_turbo
mkdir -p /root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-32B
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --local-dir /root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-32B \
    --max-workers 8
mkdir -p /root/autodl-tmp/models/Qwen3-8B/
HF_ENDPOINT=https://hf-mirror.com hf download Qwen/Qwen3-8B --local-dir /root/autodl-tmp/models/Qwen3-8B/

mkdir -p /root/autodl-tmp/models/Qwen2.5-32B-Instruct/
HF_ENDPOINT=https://hf-mirror.com hf download Qwen/Qwen2.5-32B-Instruct --local-dir /root/autodl-tmp/models/Qwen2.5-32B-Instruct/
"""
vLLM smoke inference for Qwen3-8B.

Usage (recommend verl3 which has vllm):
  source /root/autodl-tmp/miniconda3/bin/activate verl3
  CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/demo/vllm_Qwen3-8B.py
"""

from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_PATH = Path("/root/autodl-tmp/models/Qwen3-8B")

PROMPTS = [
    "1+1=",
    "用一句话介绍什么是强化学习。",
    "Solve: If x + 3 = 10, what is x? Brief answer only.",
]


def build_chat_prompts(tokenizer: AutoTokenizer, user_texts: list[str]) -> list[str]:
    formatted: list[str] = []
    for text in user_texts:
        messages = [{"role": "user", "content": text}]
        formatted.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # smoke: disable Qwen3 thinking mode
            )
        )
    return formatted


def main() -> None:
    assert MODEL_PATH.exists(), f"missing model: {MODEL_PATH}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    prompts = build_chat_prompts(tokenizer, PROMPTS)

    llm = LLM(
        model=str(MODEL_PATH),
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.4,
        enforce_eager=True,  # faster startup for smoke
    )
    sampling = SamplingParams(
        temperature=1.5,
        max_tokens=1024,
        top_p=0.95,
        top_k=50,
        n=8,
    )

    outputs = llm.generate(prompts, sampling)
    print("=" * 60)
    for user, out in zip(PROMPTS, outputs):
        text = out.outputs[0].text
        print(f"[user] {user}")
        print(f"[asst] {text}")
        print("-" * 60)
    print("SMOKE_OK")


if __name__ == "__main__":
    main()

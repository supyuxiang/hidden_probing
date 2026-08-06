"""
vLLM sampling for local Llama3.2-8B (base, no chat_template).

Usage:
  source /root/autodl-tmp/miniconda3/bin/activate verl3
  CUDA_VISIBLE_DEVICES=0 python /root/hidden_prob/demo/vllm_Llama3.2-8B.py

Notes:
  - This checkpoint has no tokenizer chat_template, so we format prompts manually.
  - Base models follow instructions weaker than Instruct; for math multilingual
    sampling like exp1_math, prefer Llama-3.1-8B-Instruct if available.
"""

from __future__ import annotations

from pathlib import Path

from vllm import LLM, SamplingParams

MODEL_PATH = Path("/root/autodl-tmp/models/Llama3.2-8B")

# Llama 3 / 3.1 Instruct-style special tokens (also commonly used as a
# prompt wrapper for base models when no chat_template is shipped).
BOS = "<|begin_of_text|>"
START = "<|start_header_id|>"
END = "<|end_header_id|>"
EOT = "<|eot_id|>"


def format_llama3_chat(
    user: str,
    system: str = "You are a helpful math assistant. Answer clearly.",
) -> str:
    """Manual chat formatting when tokenizer.chat_template is missing."""
    return (
        f"{BOS}"
        f"{START}system{END}\n\n{system}{EOT}"
        f"{START}user{END}\n\n{user}{EOT}"
        f"{START}assistant{END}\n\n"
    )


def format_completion(user: str) -> str:
    """Plain continuation-style prompt (more natural for base LMs)."""
    return f"{BOS}Problem:\n{user}\n\nSolution:\n"


PROMPTS_USER = [
    "1+1=",
    "If x + 3 = 10, what is x? Give a brief answer.",
]


def main() -> None:
    assert MODEL_PATH.exists(), f"missing model: {MODEL_PATH}"

    # mode: "chat" | "completion"
    mode = "chat"
    if mode == "chat":
        prompts = [format_llama3_chat(u) for u in PROMPTS_USER]
    else:
        prompts = [format_completion(u) for u in PROMPTS_USER]

    llm = LLM(
        model=str(MODEL_PATH),
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.4,
        enforce_eager=True,
    )
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        # stop at turn boundary if the model emits it
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )

    outputs = llm.generate(prompts, sp)
    print("=" * 60)
    for user, out in zip(PROMPTS_USER, outputs):
        print(f"[user] {user}")
        print(f"[asst] {out.outputs[0].text}")
        print("-" * 60)
    print("SMOKE_OK")


if __name__ == "__main__":
    main()

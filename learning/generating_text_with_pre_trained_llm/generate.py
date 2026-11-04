from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from safetensors.torch import load_file

from base_model.qwen import (
    QWEN_CONFIG_06_B,
    Qwen3Model,
    Qwen3Tokenizer,
    generate_text,
    load_hf_weights_into_qwen,
)


def generate_token_streams(prompt):
    model_dir = Path(__file__).resolve().parent / "qwen"
    tokenizer = Qwen3Tokenizer(model_dir / "tokenizer.json")
    weights = load_file(model_dir / "model.safetensors")

    model = Qwen3Model(QWEN_CONFIG_06_B)
    load_hf_weights_into_qwen(
        model,
        param_config={
            "n_layers": QWEN_CONFIG_06_B["n_layers"],
            "hidden_dim": QWEN_CONFIG_06_B["hidden_dim"],
        },
        params=weights,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.to(torch.bfloat16)

    response = generate_text(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=128,
        temperature=0,
        chat=True,
        enable_thinking=False,
    )
    print(response)


if __name__ == "__main__":
    generate_token_streams("what is ai")

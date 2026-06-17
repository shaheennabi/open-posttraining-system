from pathlib import Path
import sys

ROOT_DIR = Path.cwd().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from downloading_the_base_model.download_model import download_model

import torch
from safetensors.torch import load_file

from base_model.qwen import (
    QWEN_CONFIG_06_B,
    Qwen3Model,
    Qwen3Tokenizer,
    load_hf_weights_into_qwen,
)

model_dir = Path.cwd() / "qwen"


def load_model_and_tokenizer(
    which_model,
    use_compile,
    load_rlvr_checkpoint=False,
):

    if load_rlvr_checkpoint:
        download_model(
            "devshaheen/qwen3.5_0.6B_rlvr_grpo_checkpoints",
            "qwen",
        )
    else:
        download_model(
            "Qwen/Qwen3-0.6B",
            "qwen",
        )

    if which_model == "base":
        tokenizer = Qwen3Tokenizer(
            model_dir / "tokenizer.json",
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=False,
        )

    elif which_model == "reasoning":
        tokenizer = Qwen3Tokenizer(
            model_dir / "tokenizer.json",
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=True,
        )

    else:
        raise ValueError("Not a valid model type")

    model = Qwen3Model(QWEN_CONFIG_06_B)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if load_rlvr_checkpoint:

        checkpoint_path = (
            model_dir /
            "qwen3-0.6B-rlvr-grpo-step00050.safetensors"
        )

        print(f"\nLoading RLVR checkpoint: {checkpoint_path}")

        state_dict = load_file(str(checkpoint_path))

        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=False,
        )

        print(f"Loaded {len(state_dict)} tensors")
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")

    else:

        print("\nLoading base Qwen weights")

        weights = load_file(
            str(model_dir / "model.safetensors")
        )

        load_hf_weights_into_qwen(
            model,
            param_config={
                "n_layers": QWEN_CONFIG_06_B["n_layers"],
                "hidden_dim": QWEN_CONFIG_06_B["hidden_dim"],
            },
            params=weights,
        )

    model.to(device)
    model.to(torch.bfloat16)

    if use_compile:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)

    return model, tokenizer
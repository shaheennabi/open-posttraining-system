from pathlib import Path
import sys

ROOT_DIR = Path.cwd().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import torch
from base_model.qwen import Qwen3Tokenizer

from downloading_the_base_model.download_model import download_model

model_dir = Path.cwd() / "qwen"






def load_tokenizer_only(which_model):

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

    return tokenizer
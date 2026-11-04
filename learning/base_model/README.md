# Base model (Qwen)

## Overview

This folder implements a local Qwen-style model and tokenizer used across the project. The primary implementation is in `qwen.py` and exposes:

- `Qwen3Model` — model class (defined in `qwen.py`)
- `Qwen3Tokenizer` — tokenizer wrapper that loads a `tokenizer.json` file (uses the `tokenizers` library)
- `load_hf_weights_into_qwen(model, param_config, params)` — helper that copies tensors (e.g. from `safetensors`) into the model
- `QWEN_CONFIG_06_B` — a ready-made config for the 0.6B variant

This repository includes a helper script to download model artifacts from Hugging Face (optional): [downloading_the_base_model/download_model.py](downloading_the_base_model/download_model.py).

## Files

- [base_model/qwen.py](base_model/qwen.py) — implementation of `Qwen3Model`, `Qwen3Tokenizer`, KV cache, generation helpers and the weight-loading utility.
- [base_model/qwen.ipynb](base_model/qwen.ipynb) — notebook with examples and demonstrations.

## Quick start (recommended)

1. Download model artifacts (config, `model.safetensors`, `tokenizer.json`) into a local folder (for example `qwen/`). The included script is optional and uses `huggingface_hub`:

```bash
python downloading_the_base_model/download_model.py --repo-id Qwen/Qwen3-0.6B --local-dir qwen
```

2. Load tokenizer, create the model, and load weights using the local loader:

```python
from base_model.qwen import (
    Qwen3Model,
    Qwen3Tokenizer,
    load_hf_weights_into_qwen,
    QWEN_CONFIG_06_B,
)
from safetensors.torch import load_file
import json

# 1) Model config (a minimal runtime config is provided as QWEN_CONFIG_06_B)
model_cfg = QWEN_CONFIG_06_B

# 2) Instantiate model
model = Qwen3Model(model_cfg)

# 3) Tokenizer (this expects a tokenizer.json file)
tokenizer = Qwen3Tokenizer("qwen/tokenizer.json")

# 4) Load parameter layout/config and safetensors weights (downloaded into `qwen/`)
param_config = json.load(open("qwen/config.json", "r", encoding="utf-8"))
params = load_file("qwen/model.safetensors")

# 5) Copy safetensors parameters into the model
load_hf_weights_into_qwen(model, param_config, params)

# 6) Ready for inference
model.eval()
```

Notes:
- The project `Qwen3Tokenizer` uses the `tokenizers` library and reads a `tokenizer.json` file; it is not `transformers.AutoTokenizer`.
- `load_hf_weights_into_qwen` is the project's loader: it expects a `param_config` (the original model config / parameter layout) and a mapping-like `params` object (for example returned by `safetensors.torch.load_file`). It performs in-place copies and supports multiple attribute-name variants (e.g., `emb` vs `tok_emb`, `t_block` vs `trf_blocks`).
- If `lm_head.weight` is not present in the weights, the loader will tie the output head to the embedding weights and print a notice.

## Generation helper

`qwen.py` includes `generate_text(model, tokenizer, prompt, ...)` and a small sampling helper `sample_next_token(...)`. Use these for quick local inference experiments.

## Common pitfalls

- Tokenizer file missing: `Qwen3Tokenizer` requires `tokenizer.json` present at the path you pass it.
- Config/param mismatch: `param_config` must match the layout expected by `load_hf_weights_into_qwen` (e.g., `n_layers`, `num_experts` when applicable).
- Device/dtype: the model config may use lower precision (`torch.bfloat16`) — move the model to the appropriate device/dtype before large runs.

## Where to look next

- Examples and interactive exploration: [base_model/qwen.ipynb](base_model/qwen.ipynb)
- Download helper: [downloading_the_base_model/download_model.py](downloading_the_base_model/download_model.py)
- Generation and evaluation pipelines: `generating_text_with_pre_trained_llm/` and `evaluating_reasoning_models/`

If you want, I can also update any examples that still reference `transformers.AutoTokenizer` or `model.from_pretrained()` to use the local API shown above.

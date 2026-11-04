# Generating text with a pre-trained LLM

This folder shows the simple local flow for text generation using `Qwen3Model` and `Qwen3Tokenizer` from `base_model/qwen.py`.

## What this folder does

- Uses local artifacts in `generating_text_with_pre_trained_llm/qwen/`.
- Loads `tokenizer.json` with `Qwen3Tokenizer`.
- Loads `model.safetensors` with `safetensors.torch.load_file`.
- Instantiates `Qwen3Model(QWEN_CONFIG_06_B)` and copies weights with `load_hf_weights_into_qwen`.
- Runs generation through `generate_text(...)`.

## Quick usage

Place `tokenizer.json`, `model.safetensors`, and `config.json` inside `generating_text_with_pre_trained_llm/qwen/`, then run:

```bash
python generate.py
```

`generate.py` loads the local tokenizer and weights, moves the model to the available device, and prints a generated response.

## Notebook

`main.ipynb` contains a short example of the same flow: load artifacts, instantiate model and tokenizer, load weights, and generate text.

Keep it simple: no `top_k`, no `top_p`, no repetition penalty, and no unrelated transformer helper examples.

## Notes

- This repo uses a local loader, not `transformers.AutoTokenizer`.
- The model is created with a local config and weights are copied in-place.
- The notebook and script are both local examples, not full `transformers` pipelines.

# Downloading the Base Model

## Overview

This folder contains utilities and scripts for downloading pre-trained language models from the Hugging Face Model Hub. Before you can use any model for inference, evaluation, or fine-tuning, you need to download its weights and configuration files.

**Key responsibility:** Download and cache model weights from Hugging Face Hub for use in your custom architecture and subsequent tasks.

## Contents

### `download_model.py`
The main script for downloading models from Hugging Face Hub.

**Key functions:**
- Model downloading from Hugging Face
- Local caching management
- Tokenizer downloading
- Configuration file retrieval
- Bandwidth and storage optimization

**Usage:**
```bash
python download_model.py --model "qwen/Qwen-1.8B" --output_dir "./models"
```

## Workflow

### Step 1: Choose Your Model

Popular open-source models available on Hugging Face:
- **Qwen Series**: `Qwen/Qwen-1.8B`, `Qwen/Qwen-7B`, `Qwen/Qwen-14B`
- **LLaMA Series**: `meta-llama/Llama-2-7b`, `meta-llama/Llama-2-13b`
- **Mistral Series**: `mistralai/Mistral-7B`, `mistralai/Mistral-7B-Instruct`
- **Other**: MPT, Falcon, OPT, and many more

### Step 2: Configure Download Settings

Key parameters:
- `model_name` - Hugging Face model identifier (e.g., "Qwen/Qwen-7B")
- `cache_dir` - Where to save downloaded models locally
- `local_files_only` - Use only cached files (don't download)
- `force_download` - Re-download even if cached

### Step 3: Run Download Script

```bash
# Basic download
python download_model.py

# With custom output directory
python download_model.py --output_dir "/path/to/models"

# Download specific model
python download_model.py --model "qwen/Qwen-7B"
```

### Step 4: Verify Download

After downloading, verify files exist:
```bash
# List downloaded files
ls ./models/Qwen/Qwen-7B/

# Expected files:
# - pytorch_model.bin (or .safetensors) - Model weights
# - config.json - Model configuration
# - tokenizer.json or tokenizer.model - Tokenizer
# - special_tokens_map.json - Special token definitions
# - generation_config.json - Generation defaults
```

## What Gets Downloaded

### Model Weights (Required)
- **pytorch_model.bin** or **model.safetensors** - Contains all learned parameters
- **Typical size**: Varies by model (1.8B: ~3.5GB, 7B: ~14GB, 13B+: ~26GB+)

### Configuration Files (Required)
- **config.json** - Model architecture specifications
- **generation_config.json** - Default generation parameters
- **Defines**: hidden size, number of layers, attention heads, etc.

### Tokenizer Files (Required)
- **tokenizer.json** or **tokenizer.model** - Text encoding/decoding
- **tokenizer_config.json** - Tokenizer settings
- **Defines**: vocabulary, special tokens, encoding rules

### Optional Files
- **README.md** - Model documentation
- **Model card** - Model description and usage guidelines
- **Quantization files** - Pre-quantized versions (smaller, faster)

## Caching Strategy

Hugging Face automatically caches downloads in:
- **Linux/Mac:** `~/.cache/huggingface/hub/`
- **Windows:** `C:\Users\<username>\.cache\huggingface\hub\`

### Benefits of Caching
- **Avoid re-downloading** - Same model used in multiple projects
- **Save bandwidth** - Don't re-download on reinstall
- **Faster initialization** - Subsequent loads are instant

### Manage Cache
```bash
# View cache size
du -sh ~/.cache/huggingface/hub/

# Clear entire cache (⚠️ will re-download next use)
rm -rf ~/.cache/huggingface/hub/
```

## Models for Different Use Cases

### Text Generation & Inference
- Qwen (1.8B, 7B, 14B, 72B variants)
- LLaMA 2 (7B, 13B, 70B)
- Mistral 7B
- Falcon 7B

### Fine-tuning & Training
- Smaller models (1.8B-7B) for development
- Larger models (13B+) for better quality results

### Evaluation & Benchmarking
- Same models used throughout the project
- Ensures consistency across experiments

### Resource-Constrained Settings
- 1.8B models - Laptops, consumer GPUs
- 7B models - Gaming GPUs (RTX 3090, etc.)
- 13B+ models - High-end GPUs or multi-GPU setups

## Troubleshooting

### Issue: Download is Slow
**Solution:**
- Check internet connection speed
- Try downloading during off-peak hours
- Verify you have sufficient disk space

### Issue: Not Enough Disk Space
**Problem:** Model weights are large (3-70GB depending on size)
**Solution:**
- Check available disk space: `df -h`
- Remove old cached models if needed
- Use external drives for model storage
- Consider smaller model variants

### Issue: Authentication Required
**Problem:** Some models require Hugging Face login
**Solution:**
```bash
# Login with your Hugging Face token
huggingface-cli login

# Enter your token when prompted
# Token available at: https://huggingface.co/settings/tokens
```

### Issue: Corrupted Download
**Problem:** Downloaded file incomplete or corrupted
**Solution:**
```bash
# Force re-download
python download_model.py --force_download --model "your/model"
```

## Best Practices

1. **Download once, use many times** - Cache allows sharing across projects
2. **Verify model size** - Check disk space before downloading
3. **Use appropriate model size** - Balance quality and hardware constraints
4. **Document model versions** - Track which model/version used in experiments
5. **Keep local copies** - Back up important model weights

## Next Steps After Downloading

Once you've downloaded a model:

1. **Set up architecture** → Go to `../base_model/`
   - Load weights into your custom architecture
   - Initialize tokenizer

2. **Test generation** → Go to `../generating_text_with_pre_trained_llm/`
   - Generate text to verify model works
   - Test different sampling strategies

3. **Evaluate performance** → Go to `../evaluating_reasoning_models/`
   - Benchmark on reasoning tasks
   - Measure baseline accuracy

4. **Improve reasoning** → Go to `../improving_reasoning_with_inference_time_scaling/`
   - Apply inference-time optimizations
   - Compare improvement over baseline

## Integration with Other Folders

```
Downloading Base Model (YOU ARE HERE)
         ↓
    Downloads weights from Hugging Face
         ↓
Used by:
- base_model/ (load into architecture)
- generating_text_with_pre_trained_llm/ (inference)
- evaluating_reasoning_models/ (testing)
- improving_reasoning_with_inference_time_scaling/ (optimization)
```

## Reference

- [Hugging Face Model Hub](https://huggingface.co/models)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [Available Open Models Comparison](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

## Questions?

- Check the Hugging Face documentation
- Review `download_model.py` script comments
- Refer to main README for project overview

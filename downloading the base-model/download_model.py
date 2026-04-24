import os
from huggingface_hub import snapshot_download
from pathlib import Path
## downloading the model from huggingface

def download_model(repo_id, local_dir):
    

    if os.path.exists(local_dir):
        downloaded_path = snapshot_download(repo_id=repo_id, local_dir=local_dir, allow_patterns=["*config.json", "*model.safetensors", "*tokenizer.json", "*tokenizer-config.json"], ignore_patterns=["vocab.json", "merges.txt", "generation_config.json", "readme.md", "LICENSE", ".gitattributes"])
    else:
        local_dir = os.makedirs(local_dir)
        downloaded_path = snapshot_download(repo_id=repo_id, local_dir=local_dir, allow_patterns=["*config.json", "*model.safetensors", "*tokenizer.json", "*tokenizer-config.json"], ignore_patterns=["vocab.json", "merges.txt", "generation_config.json", "readme.md", "LICENSE", ".gitattributes"])

    return downloaded_path


download_model(repo_id="Qwen/Qwen3-0.6B-Base", local_dir="qwen")
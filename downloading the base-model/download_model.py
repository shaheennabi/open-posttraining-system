import os
from huggingface_hub import snapshot_download

def downlaod_model(repo_id, local_dir):
    os.makedirs(local_dir, exist_ok=True)


    downlaod_path = snapshot_download(repo_id=repo_id,
                    local_dir=local_dir,
                    allow_patterns=[
                        "config.json",
                        "model.safetensors",
                        "tokenizer.json",
                        "tokenizer_config.json",
                    ])
    return downlaod_model



downlaod_model(
    repo_id="Qwen/Qwen3-0.6B-Base",
    local_dir="qwen",
    )
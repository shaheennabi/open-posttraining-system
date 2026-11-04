import os
from huggingface_hub import snapshot_download

def download_model(repo_id="Qwen/Qwen3-0.6B", local_dir="qwen"):
    os.makedirs(local_dir, exist_ok=True)

    downlaod_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            #"qwen3-0.6B-rlvr-grpo-step00005.safetensors",
            "qwen3-0.6B-rlvr-grpo-step00050.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
    )
    return downlaod_path

if __name__ == "__main__":
    download_model()

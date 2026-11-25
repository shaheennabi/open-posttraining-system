from pathlib import Path

from huggingface_hub import snapshot_download


def download_model(repo_id="Qwen/Qwen3-0.6B", local_dir=None):
    if local_dir is None:
        local_dir = Path(__file__).resolve().parent / "qwen"
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    download_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
    )
    return download_path

if __name__ == "__main__":
    download_model()

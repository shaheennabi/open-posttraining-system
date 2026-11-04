from datasets import load_dataset
from pathlib import Path
import json
import requests


def save_math500_test(output_path="math500_test.json"):
    """Download MATH-500 from Hugging Face and save locally."""
    dset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math_data = dset.to_list()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(math_data, f, ensure_ascii=False, indent=2)

    return math_data


def load_math500_test(local_path="math500_test.json", save_copy=True):
    """Load local copy if present, otherwise download from GitHub."""
    local_path = Path(local_path)

    url = (
        "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
        "main/ch03/01_main-chapter-code/math500_test.json"
    )

    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()

        if save_copy:
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    return data
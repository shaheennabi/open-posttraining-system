"""Small utility to download and save the MATH-500 test set into this repo.

Usage:
    python scripts/download_math500.py --output evaluating_reasoning_models/math500_test.json
"""
from pathlib import Path
import argparse
import json


def download_math500(output: str | None = None):
    """Download MATH-500 and save to `output`. Returns the loaded data list.

    If `output` is None, saves to the parent `research` folder next to this script
    as `math500.json` (i.e. `.../research/math500.json`).
    """
    if output is None:
        repo_research_dir = Path(__file__).resolve().parent.parent
        out_path = repo_research_dir / "math500.json"
    else:
        out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Import and use the project's helper if available
    try:
        from evaluating_reasoning_models.load_math_500 import save_math500_test

        data = save_math500_test(output_path=str(out_path))
        return data
    except Exception:
        # Fallback: try the GitHub raw URL
        import requests

        url = (
            "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
            "main/ch03/01_main-chapter-code/math500_test.json"
        )
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data


def main(output: str = None):
    parser = argparse.ArgumentParser()
    default_output = str(Path(__file__).resolve().parent.parent / "math500.json")
    parser.add_argument("--output", default=default_output)
    args = parser.parse_args([] if output is None else ["--output", output])
    data = download_math500(output=args.output)
    print(f"Saved {len(data)} examples to {args.output}")


if __name__ == "__main__":
    main()

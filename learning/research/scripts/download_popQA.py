"""Download PopQA from Hugging Face and save it locally."""
from pathlib import Path
import argparse
import json


def download_popQA(output: str | None = None):
    """Download PopQA and save to `output`. Returns the loaded data list.

    If `output` is None, saves to the parent `research` folder next to this script
    as `popqa.json`.
    """
    if output is None:
        repo_research_dir = Path(__file__).resolve().parent.parent
        out_path = repo_research_dir / "popqa.json"
    else:
        out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        dset = load_dataset("akariasai/PopQA")
        # If a default split exists, use it; otherwise collect all splits.
        if isinstance(dset, dict):
            if "train" in dset:
                data = dset["train"].to_list()
            else:
                data = []
                for split in dset.values():
                    data.extend(split.to_list())
        else:
            data = dset.to_list()

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data
    except Exception as e:
        raise RuntimeError(
            "Failed to download PopQA. Make sure the `datasets` package is installed and the dataset name is correct."
        ) from e


def main(output: str = None):
    parser = argparse.ArgumentParser()
    default_output = str(Path(__file__).resolve().parent.parent / "popqa.json")
    parser.add_argument("--output", default=default_output)
    args = parser.parse_args([] if output is None else ["--output", output])
    data = download_popQA(output=args.output)
    print(f"Saved {len(data)} examples to {args.output}")


if __name__ == "__main__":
    main()

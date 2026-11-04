from pathlib import Path


def from_root() -> str:
    """Return the repository root directory as a string."""
    return str(Path(__file__).resolve().parent)

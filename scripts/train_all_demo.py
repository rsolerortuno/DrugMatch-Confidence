"""Train all lightweight bundled demonstrations from the repository root."""

import json
from dataclasses import asdict
from pathlib import Path

from drugmatch.workflows import run_synthetic_training


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    outcomes = run_synthetic_training(root)
    print(json.dumps([asdict(item) for item in outcomes], indent=2))

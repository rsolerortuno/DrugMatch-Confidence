"""Fail-fast verification of files and bundled model artifacts required for release."""

from __future__ import annotations

import json
from pathlib import Path

from drugmatch.api import DrugMatchPredictor


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "pyproject.toml",
    "Dockerfile",
    ".github/workflows/ci.yml",
    "docs/ARCHITECTURE.md",
    "docs/DATA.md",
    "docs/BIOLOGICAL_RATIONALE.md",
    "docs/REAL_DATA_TRAINING.md",
    "reports/model_card.md",
    "reports/data_audit/prism_drug_audit.csv",
    "reports/examples/example_input.csv",
]


def main() -> None:
    missing = [path for path in REQUIRED if not (ROOT / path).exists()]
    if missing:
        raise SystemExit(f"Missing release files: {missing}")
    groups = ["synthetic_omics", "prism_lineage"]
    records: list[dict] = []
    for group in groups:
        model_paths = sorted((ROOT / "models" / "demo" / group).glob("*.joblib"))
        if len(model_paths) != 5:
            raise SystemExit(f"Expected five {group} models; found {len(model_paths)}")
        for path in model_paths:
            predictor = DrugMatchPredictor.load(path)
            reference = predictor.bundle["training_reference"].iloc[[0]]
            result = predictor.predict(reference)
            if "preclinical" not in result.disclaimer.lower():
                raise SystemExit(f"Missing preclinical disclaimer in {path}")
            records.append(
                {
                    "group": group,
                    "drug": result.drug,
                    "model": str(path.relative_to(ROOT)),
                    "feature_coverage": result.feature_coverage,
                    "prediction_class": result.predicted_class,
                }
            )
    output = ROOT / "reports" / "release_check.json"
    output.write_text(json.dumps({"status": "pass", "models": records}, indent=2), encoding="utf-8")
    print(f"Release check passed for {len(records)} bundled model artifacts")


if __name__ == "__main__":
    main()

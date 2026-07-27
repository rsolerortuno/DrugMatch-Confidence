"""Generate lightweight, reproducible reports from bundled model artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from drugmatch.api import DrugMatchPredictor


ROOT = Path(__file__).resolve().parents[1]


def collect(group: str) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted((ROOT / "models" / "demo" / group).glob("*_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        drug = path.name.removesuffix("_metrics.json")
        row = {
            "dataset": group,
            "drug": drug,
            "regression_mae": payload["regression"]["mae"],
            "regression_rmse": payload["regression"]["rmse"],
            "regression_r2": payload["regression"]["r2"],
            "regression_spearman": payload["regression"]["spearman"],
            "classification_auroc": payload["classification"]["auroc"],
            "classification_auprc": payload["classification"]["auprc"],
            "classification_balanced_accuracy": payload["classification"]["balanced_accuracy"],
            "classification_brier": payload["classification"]["brier"],
            "interval_coverage": payload["interval_coverage"],
            "decision_threshold": payload["decision_threshold"],
            "calibration_method": payload["calibration_method"],
        }
        for name, metrics in payload["baselines"].items():
            for metric, value in metrics.items():
                row[f"baseline__{name}__{metric}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def bar_plot(frame: pd.DataFrame, column: str, title: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    frame.sort_values(column).plot.barh(x="drug", y=column, ax=axis, legend=False)
    axis.set_title(title)
    axis.set_xlabel(column.replace("_", " ").title())
    axis.set_ylabel("Drug")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def create_prediction_example() -> None:
    model_path = ROOT / "models" / "demo" / "synthetic_omics" / "trametinib.joblib"
    predictor = DrugMatchPredictor.load(model_path)
    reference = predictor.bundle["training_reference"]
    sample = reference.iloc[[0]].copy()
    example_dir = ROOT / "reports" / "examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(example_dir / "example_input.csv", index_label="model_id")
    result = predictor.predict(sample)
    (example_dir / "example_prediction.json").write_text(
        json.dumps(asdict(result), indent=2), encoding="utf-8"
    )


def main() -> None:
    internal = ROOT / "reports" / "internal_validation"
    internal.mkdir(parents=True, exist_ok=True)
    figures = ROOT / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    synthetic = collect("synthetic_omics")
    prism = collect("prism_lineage")
    synthetic.to_csv(internal / "synthetic_omics_metrics.csv", index=False)
    prism.to_csv(internal / "prism_lineage_metrics.csv", index=False)
    pd.concat([synthetic, prism], ignore_index=True).to_csv(
        internal / "all_demo_metrics.csv", index=False
    )

    bar_plot(
        synthetic,
        "regression_r2",
        "Synthetic full-omics demonstration: XGBoost regression",
        figures / "synthetic_regression_r2.png",
    )
    bar_plot(
        synthetic,
        "classification_auroc",
        "Synthetic full-omics demonstration: XGBoost classification",
        figures / "synthetic_classification_auroc.png",
    )
    bar_plot(
        prism,
        "regression_spearman",
        "Real PRISM lineage-only demonstration",
        figures / "prism_lineage_spearman.png",
    )
    create_prediction_example()

    summary = """# Bundled demonstration summary

The repository contains two deliberately different demonstrations:

1. **Synthetic full-omics models.** These validate the complete expression, mutation,
   copy-number, lineage, XGBoost, calibration, conformal interval, OOD and SHAP pipeline.
   Their performance must not be interpreted as biological evidence because the data were
   generated with known rules.
2. **Real PRISM lineage-only models.** These validate real response ingestion and provide an
   honest baseline. Trametinib and afatinib show lineage-associated signal, whereas
   palbociclib, olaparib and gemcitabine are poorly explained by lineage alone. This supports
   the need for the full DepMap molecular features rather than claiming that tissue type is
   sufficient.

The final full-molecular real-data models are intentionally not misrepresented as complete in
this bundle. The code is ready to train them after the four documented DepMap public-release
files are placed in `data/raw/depmap/`.
"""
    (ROOT / "reports" / "DEMO_RESULTS.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()

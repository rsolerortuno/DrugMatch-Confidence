"""Generate bounded-compute robustness reports using the deterministic synthetic cohort."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from drugmatch.api import DrugMatchPredictor
from drugmatch.data import load_model_metadata, load_wide_omics
from drugmatch.external import validate_external
from drugmatch.preprocessing import OmicsTables, assemble_features
from drugmatch.robustness import feature_stability, leave_one_lineage_out, modality_ablation
from drugmatch.synthetic import generate_synthetic_omics

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {"n_estimators": 50, "n_jobs": 2}


def load_cohort(directory: Path):
    return (
        assemble_features(
            OmicsTables(
                load_wide_omics(directory / "expression.csv"),
                load_wide_omics(directory / "mutations.csv"),
                load_wide_omics(directory / "copy_number.csv"),
                load_model_metadata(directory / "metadata.csv"),
            )
        ),
        pd.read_csv(directory / "response.csv"),
        load_model_metadata(directory / "metadata.csv"),
    )


def main() -> None:
    cohort = ROOT / "data" / "processed" / "synthetic_omics"
    features, response, metadata = load_cohort(cohort)
    interpretation = ROOT / "reports" / "interpretation"
    internal = ROOT / "reports" / "internal_validation" / "lineage_holdout"
    external = ROOT / "reports" / "external_validation"
    interpretation.mkdir(parents=True, exist_ok=True)
    internal.mkdir(parents=True, exist_ok=True)
    external.mkdir(parents=True, exist_ok=True)

    modality_ablation(features, response, metadata, "trametinib", model_params=PARAMS).to_csv(
        interpretation / "trametinib_synthetic_modality_ablation.csv", index=False
    )
    feature_stability(
        features, response, "trametinib", n_splits=3, top_n=20, model_params=PARAMS
    ).to_csv(interpretation / "trametinib_synthetic_feature_stability.csv", index=False)
    leave_one_lineage_out(
        features,
        response,
        metadata,
        "trametinib",
        min_lineage_models=50,
        model_params=PARAMS,
    ).to_csv(internal / "trametinib_synthetic_lineage_holdout.csv", index=False)

    # Independent synthetic cohort: useful only for testing frozen external-validation code.
    external_dir = ROOT / "data" / "processed" / "synthetic_external"
    generate_synthetic_omics(external_dir, n_models=300, n_noise_genes=120, seed=99)
    external_features, external_response, _ = load_cohort(external_dir)
    predictor = DrugMatchPredictor.load(
        ROOT / "models" / "demo" / "synthetic_omics" / "trametinib.joblib"
    )
    standardized = external_response[external_response["drug"].eq("trametinib")][
        ["model_id", "auc"]
    ].rename(columns={"auc": "external_response"})
    metrics, predictions = validate_external(predictor, external_features, standardized)
    predictions.to_csv(external / "synthetic_trametinib_predictions.csv", index=False)
    (external / "synthetic_trametinib_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

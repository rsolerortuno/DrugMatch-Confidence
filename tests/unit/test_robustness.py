from pathlib import Path

import pandas as pd

from drugmatch.data import load_model_metadata, load_wide_omics
from drugmatch.preprocessing import OmicsTables, assemble_features
from drugmatch.robustness import feature_stability, leave_one_lineage_out, modality_ablation
from drugmatch.synthetic import generate_synthetic_omics


def _data(tmp_path: Path):
    paths = generate_synthetic_omics(tmp_path, n_models=180, n_noise_genes=20, seed=13)
    tables = OmicsTables(
        load_wide_omics(paths["expression"]),
        load_wide_omics(paths["mutations"]),
        load_wide_omics(paths["copy_number"]),
        load_model_metadata(paths["metadata"]),
    )
    return assemble_features(tables), pd.read_csv(paths["response"]), tables.metadata


def test_modality_ablation_and_stability(tmp_path: Path) -> None:
    features, response, metadata = _data(tmp_path)
    params = {"n_estimators": 10, "n_jobs": 1}
    ablation = modality_ablation(features, response, metadata, "trametinib", model_params=params)
    stability = feature_stability(features, response, "trametinib", n_splits=2, top_n=5, model_params=params)
    assert "all_modalities" in set(ablation["ablation"])
    assert not stability.empty
    assert stability["frequency_fraction"].between(0, 1).all()


def test_lineage_holdout_returns_schema(tmp_path: Path) -> None:
    features, response, metadata = _data(tmp_path)
    result = leave_one_lineage_out(
        features,
        response,
        metadata,
        "trametinib",
        min_lineage_models=20,
        model_params={"n_estimators": 10, "n_jobs": 1},
    )
    assert set(["held_out_lineage", "regression_spearman", "classification_auroc"]).issubset(result.columns)

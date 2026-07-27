from pathlib import Path

import pandas as pd

from drugmatch.data import load_model_metadata, load_wide_omics
from drugmatch.preprocessing import OmicsTables, assemble_features
from drugmatch.reporting import generate_internal_figures
from drugmatch.synthetic import generate_synthetic_omics
from drugmatch.training import train_drug_bundle


def test_internal_reporting_creates_roc_and_accuracy_figures(tmp_path: Path) -> None:
    paths = generate_synthetic_omics(tmp_path / "data", n_models=220, n_noise_genes=10, seed=19)
    metadata = load_model_metadata(paths["metadata"])
    features = assemble_features(
        OmicsTables(
            load_wide_omics(paths["expression"]),
            load_wide_omics(paths["mutations"]),
            load_wide_omics(paths["copy_number"]),
            metadata,
        )
    )
    response = pd.read_csv(paths["response"])
    model_dir = tmp_path / "models"
    train_drug_bundle(
        "trametinib",
        features,
        response,
        metadata,
        model_dir,
        data_release="test",
        regression_params={"n_estimators": 10, "n_jobs": 1},
        classification_params={"n_estimators": 10, "n_jobs": 1},
        seed=19,
    )
    output = tmp_path / "figures"
    generate_internal_figures(model_dir, output, ["trametinib"])
    assert (output / "internal_roc_curves.png").exists()
    assert (output / "internal_accuracy_threshold_curves.png").exists()
    assert (output / "internal_performance_summary.csv").exists()

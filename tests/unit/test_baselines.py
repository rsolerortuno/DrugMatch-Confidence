from pathlib import Path

import pandas as pd

from drugmatch.baselines import evaluate_baselines
from drugmatch.data import load_model_metadata, load_wide_omics
from drugmatch.preprocessing import OmicsTables, assemble_features
from drugmatch.synthetic import generate_synthetic_omics


def test_evaluate_baselines(tmp_path: Path) -> None:
    paths = generate_synthetic_omics(tmp_path, n_models=120, n_noise_genes=10, seed=21)
    metadata = load_model_metadata(paths["metadata"])
    features = assemble_features(
        OmicsTables(
            load_wide_omics(paths["expression"]),
            load_wide_omics(paths["mutations"]),
            load_wide_omics(paths["copy_number"]),
            metadata,
        )
    )
    results = evaluate_baselines(features, pd.read_csv(paths["response"]), metadata, "trametinib")
    assert "lineage_regression" in results
    assert "elastic_net_classification" in results

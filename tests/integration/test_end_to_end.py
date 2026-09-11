from pathlib import Path

import pandas as pd
import pytest

from drugmatch.api import DrugMatchPredictor
from drugmatch.data import load_model_metadata, load_wide_omics
from drugmatch.preprocessing import OmicsTables, assemble_features
from drugmatch.synthetic import generate_synthetic_omics
from drugmatch.training import train_drug_bundle


@pytest.mark.parametrize("index_dtype", ["object", "string[pyarrow]"])
def test_end_to_end_training_and_prediction(tmp_path: Path, index_dtype: str) -> None:
    paths = generate_synthetic_omics(tmp_path / "data", n_models=140, n_noise_genes=20, seed=7)
    expression = load_wide_omics(paths["expression"])
    mutations = load_wide_omics(paths["mutations"])
    copy_number = load_wide_omics(paths["copy_number"])
    metadata = load_model_metadata(paths["metadata"])
    response = pd.read_csv(paths["response"])
    features = assemble_features(OmicsTables(expression, mutations, copy_number, metadata))
    # Exercise the Arrow-backed identifiers that broke sklearn split indexing.
    features.index = features.index.astype(index_dtype)
    metadata.index = metadata.index.astype(index_dtype)
    response["model_id"] = response["model_id"].astype(index_dtype)
    outcome = train_drug_bundle(
        "trametinib",
        features,
        response,
        metadata,
        tmp_path / "models",
        data_release="test",
        regression_params={"n_estimators": 20, "n_jobs": 1},
        classification_params={"n_estimators": 20, "n_jobs": 1},
        seed=7,
    )
    predictor = DrugMatchPredictor.load(outcome.model_path)
    result = predictor.predict(features.iloc[[0]])
    assert result.drug == "trametinib"
    assert 0 <= result.sensitivity_probability <= 1
    assert result.interval_lower < result.interval_upper
    assert result.feature_coverage == 1.0
    assert "preclinical" in result.disclaimer.lower()

    # Tuning cannot consume any calibration/test model, and imputation/feature
    # selection must use the fit membership only.
    roles = predictor.bundle["partition_roles"]
    for role, members in roles.items():
        assert len(members) == len(set(members))
        for other, other_members in roles.items():
            if role != other:
                assert not set(members) & set(other_members)
    assert set(predictor.bundle["training_reference"].index) == set(roles["fit"])
    assert result.abstain
    assert "model_evidence_not_sufficient" in result.abstention_reasons

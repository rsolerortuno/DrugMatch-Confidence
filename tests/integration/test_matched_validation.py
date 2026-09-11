import pandas as pd
import pytest

from drugmatch.data import load_model_metadata, load_wide_omics
from drugmatch.matched_validation import matched_oof
from drugmatch.preprocessing import OmicsTables, assemble_features
from drugmatch.synthetic import generate_synthetic_omics


@pytest.mark.parametrize("index_dtype", ["object", "string[pyarrow]"])
def test_same_models_and_fit_only_thresholds_for_every_family(tmp_path, index_dtype):
    paths = generate_synthetic_omics(tmp_path, n_models=140, n_noise_genes=8, seed=7)
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
    # Exercise the Arrow-backed identifiers that broke sklearn split indexing.
    features.index = features.index.astype(index_dtype)
    metadata.index = metadata.index.astype(index_dtype)
    response["model_id"] = response["model_id"].astype(index_dtype)
    calibration_rows = []
    preds, roles = matched_oof(
        features,
        response,
        metadata,
        "trametinib",
        n_splits=3,
        calibration_output=calibration_rows,
        model_params={"n_estimators": 8, "n_jobs": 1},
    )
    calibration = pd.DataFrame(calibration_rows)
    assert (calibration.absolute_residual >= 0).all()
    for fold, group in calibration.groupby("fold"):
        expected = set(roles.loc[roles.fold.eq(fold) & roles.role.eq("calibration"), "model_id"])
        for _, family in group.groupby("family"):
            assert set(family.model_id) == expected
    auc = response[response.drug.eq("trametinib")].set_index("model_id").auc
    assert set(preds.family) == {"xgboost", "elastic_net", "lineage"}
    assert not preds[["family", "model_id"]].duplicated().any()
    assert preds.groupby("family").size().nunique() == 1
    for fold, group in roles.groupby("fold"):
        sets = {role: set(rows.model_id) for role, rows in group.groupby("role")}
        assert not sets["fit"] & sets["calibration"]
        assert not sets["test"] & (sets["fit"] | sets["calibration"])
        scores = preds[preds.fold.eq(fold)]
        assert scores.sensitive_auc_max.nunique() == 1
        assert scores.sensitive_auc_max.iloc[0] == auc.loc[list(sets["fit"])].quantile(0.25)
        assert scores.resistant_auc_min.iloc[0] == auc.loc[list(sets["fit"])].quantile(0.75)

"""External validation utilities for GDSC2 and other independent response studies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from drugmatch.api import DrugMatchPredictor
from drugmatch.evaluation import classification_metrics


def map_gdsc_models(gdsc: pd.DataFrame, depmap_metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map GDSC Sanger model IDs to stable DepMap ModelIDs with an audit table."""
    metadata = depmap_metadata.copy()
    if metadata.index.name != "model_id":
        metadata.index.name = "model_id"
    if "SangerModelID" not in metadata.columns:
        raise ValueError("DepMap Model.csv must contain SangerModelID for GDSC2 mapping")
    mapping = (
        metadata.reset_index()[["model_id", "SangerModelID", "CellLineName", "COSMICID"]]
        .dropna(subset=["SangerModelID"])
        .assign(SangerModelID=lambda x: x["SangerModelID"].astype(str))
    )
    duplicates = mapping[mapping["SangerModelID"].duplicated(keep=False)].copy()
    unique_mapping = mapping.drop_duplicates("SangerModelID", keep=False)
    output = gdsc.copy()
    output["SANGER_MODEL_ID"] = output["SANGER_MODEL_ID"].astype(str)
    output = output.merge(
        unique_mapping[["SangerModelID", "model_id"]],
        left_on="SANGER_MODEL_ID",
        right_on="SangerModelID",
        how="left",
    ).drop(columns=["SangerModelID"])
    audit = pd.DataFrame(
        {
            "n_gdsc_rows": [len(gdsc)],
            "n_unique_sanger_ids": [gdsc["SANGER_MODEL_ID"].astype(str).nunique()],
            "n_mapped_rows": [int(output["model_id"].notna().sum())],
            "n_mapped_models": [int(output["model_id"].nunique())],
            "n_ambiguous_sanger_ids": [int(duplicates["SangerModelID"].nunique())],
        }
    )
    return output, audit


def standardize_external_response(
    frame: pd.DataFrame,
    drug: str,
    drug_column: str = "DRUG_NAME",
    model_column: str = "model_id",
    response_column: str = "AUC",
) -> pd.DataFrame:
    """Select and standardize one external drug-response table."""
    missing = {drug_column, model_column, response_column} - set(frame.columns)
    if missing:
        raise ValueError(f"External response table is missing columns: {sorted(missing)}")
    selected = frame[frame[drug_column].astype(str).str.lower().eq(drug.lower())].copy()
    selected = selected.dropna(subset=[model_column, response_column])
    selected = selected.rename(columns={model_column: "model_id", response_column: "external_response"})
    selected["model_id"] = selected["model_id"].astype(str)
    return selected.groupby("model_id", as_index=False)["external_response"].median()


def validate_external(
    predictor: DrugMatchPredictor,
    features: pd.DataFrame,
    external_response: pd.DataFrame,
    allowed_model_ids: list[str] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate a frozen model in batch without tuning or repeated SHAP computation."""
    response = external_response.set_index("model_id")
    common = features.index.intersection(response.index)
    if allowed_model_ids is not None:
        common = common.intersection(pd.Index(allowed_model_ids))
    if len(common) < 20:
        raise ValueError(f"External validation requires at least 20 overlapping models; found {len(common)}")
    frame = predictor._frame(features.loc[common])
    regression = predictor.bundle["regression"]
    classification = predictor.bundle["classification"]
    predicted_auc = regression.predict(frame)
    raw_probability = classification.predict_proba(frame)[:, 1]
    probability = predictor.bundle["calibrator"].predict(raw_probability)
    transformed = regression.named_steps["preprocess"].transform(frame)
    ood = predictor.bundle["ood"].label(transformed)
    predictions = pd.DataFrame(
        {
            "model_id": common,
            "predicted_auc": predicted_auc,
            "sensitivity_probability": probability,
            "external_response": response.loc[common, "external_response"].to_numpy(),
            "ood_status": ood,
        }
    )
    correlation = spearmanr(
        predictions["predicted_auc"], predictions["external_response"], nan_policy="omit"
    ).statistic
    lower = float(predictions["external_response"].quantile(0.25))
    upper = float(predictions["external_response"].quantile(0.75))
    tails = predictions[
        (predictions["external_response"] <= lower) | (predictions["external_response"] >= upper)
    ].copy()
    tails["external_sensitive"] = (tails["external_response"] <= lower).astype(int)
    threshold = float(predictor.bundle.get("decision_threshold", 0.5))
    tail_metrics = classification_metrics(
        tails["external_sensitive"].to_numpy(),
        tails["sensitivity_probability"].to_numpy(),
        threshold=threshold,
    )
    predictions["external_class"] = "intermediate"
    predictions.loc[predictions["external_response"] <= lower, "external_class"] = "sensitive"
    predictions.loc[predictions["external_response"] >= upper, "external_class"] = "resistant"
    metrics = {
        "n_overlap": float(len(predictions)),
        "n_tail_models": float(len(tails)),
        "spearman": float(correlation) if np.isfinite(correlation) else 0.0,
        "ood_fraction": float((predictions["ood_status"] == "out-of-distribution").mean()),
        "external_sensitive_auc_max": lower,
        "external_resistant_auc_min": upper,
        **{f"tail_{key}": value for key, value in tail_metrics.items()},
    }
    return metrics, predictions


def observed_assay_concordance(
    prism_response: pd.DataFrame,
    external_response: pd.DataFrame,
    drug: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Measure direct PRISM-versus-external response agreement as a transfer ceiling."""
    prism = prism_response[prism_response["drug"].astype(str).str.lower().eq(drug.lower())][
        ["model_id", "auc"]
    ].drop_duplicates("model_id")
    external = external_response.rename(columns={"external_response": "gdsc_auc"})
    merged = prism.merge(external, on="model_id", how="inner")
    if len(merged) < 20:
        return {"n_overlap": float(len(merged)), "spearman": float("nan")}, merged
    correlation = spearmanr(merged["auc"], merged["gdsc_auc"], nan_policy="omit").statistic
    return {
        "n_overlap": float(len(merged)),
        "spearman": float(correlation) if np.isfinite(correlation) else 0.0,
    }, merged

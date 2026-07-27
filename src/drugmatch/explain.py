"""Global and local XGBoost explanations using SHAP."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Extract transformed column names from a fitted scikit-learn pipeline."""
    preprocessor = pipeline.named_steps["preprocess"]
    try:
        return list(map(str, preprocessor.get_feature_names_out()))
    except Exception:
        width = int(pipeline.named_steps["model"].n_features_in_)
        return [f"feature_{index}" for index in range(width)]


def shap_values(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Calculate SHAP values for a fitted tree model."""
    transformed = pipeline.named_steps["preprocess"].transform(X)
    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[-1]
    return np.asarray(values), transformed_feature_names(pipeline)


def global_importance(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Return mean absolute SHAP importance."""
    values, names = shap_values(pipeline, X)
    importance = np.abs(values).mean(axis=0)
    return pd.DataFrame({"feature": names, "mean_abs_shap": importance}).sort_values(
        "mean_abs_shap", ascending=False
    )


def local_explanation(pipeline: Pipeline, X: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return signed local drivers for the first sample."""
    values, names = shap_values(pipeline, X.iloc[[0]])
    frame = pd.DataFrame({"feature": names, "shap_value": values[0]})
    frame["absolute_value"] = frame["shap_value"].abs()
    return frame.nlargest(top_n, "absolute_value").drop(columns="absolute_value")

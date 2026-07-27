"""Drug-response regression and classification labels."""

from __future__ import annotations

import pandas as pd


def labels_for_drug(
    response: pd.DataFrame,
    drug: str,
    sensitive_quantile: float = 0.25,
    resistant_quantile: float = 0.75,
) -> pd.DataFrame:
    """Return model-indexed continuous, percentile and categorical labels."""
    selected = response[response["drug"].str.lower().eq(drug.lower())].copy()
    selected = selected.dropna(subset=["model_id", "auc"]).drop_duplicates("model_id")
    if len(selected) < 20:
        raise ValueError(f"Drug {drug!r} has only {len(selected)} usable observations")
    lower = selected["auc"].quantile(sensitive_quantile)
    upper = selected["auc"].quantile(resistant_quantile)
    selected["response_percentile"] = selected["auc"].rank(pct=True, method="average")
    selected["class"] = "intermediate"
    selected.loc[selected["auc"] <= lower, "class"] = "sensitive"
    selected.loc[selected["auc"] >= upper, "class"] = "resistant"
    selected["binary_class"] = selected["class"].map({"sensitive": 1, "resistant": 0})
    return selected.set_index("model_id")[["auc", "response_percentile", "class", "binary_class"]]

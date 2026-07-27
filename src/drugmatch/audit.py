"""Drug-level data audit and deterministic MVP selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DrugSelectionCriteria:
    min_models: int = 350
    max_missing_fraction: float = 0.20
    min_sensitive: int = 50
    min_resistant: int = 50
    min_lineages: int = 5
    sensitive_quantile: float = 0.25
    resistant_quantile: float = 0.75


def audit_drugs(
    response: pd.DataFrame,
    metadata: pd.DataFrame,
    criteria: DrugSelectionCriteria | None = None,
) -> pd.DataFrame:
    """Summarize drug suitability using response coverage and heterogeneity."""
    criteria = criteria or DrugSelectionCriteria()
    required = {"model_id", "drug", "auc"}
    if missing := required - set(response.columns):
        raise ValueError(f"Response table is missing columns: {sorted(missing)}")
    meta = metadata.copy()
    if meta.index.name != "model_id":
        if "model_id" in meta:
            meta = meta.set_index("model_id")
        else:
            meta.index.name = "model_id"
    joined = response.merge(meta[["lineage"]], left_on="model_id", right_index=True, how="left")
    # PRISM does not encode unmeasured drug-model pairs as explicit rows. The largest
    # per-drug coverage is therefore a better estimate of the screened model universe
    # than the full metadata table, which also contains models not used in this screen.
    screened_universe = int(response.groupby("drug")["model_id"].nunique().max())
    records: list[dict] = []
    for drug, group in joined.groupby("drug", sort=True):
        total_rows = len(group)
        usable = group.dropna(subset=["auc", "model_id"]).drop_duplicates("model_id")
        n_models = usable["model_id"].nunique()
        missing_fraction = 1.0 - (n_models / max(screened_universe, 1))
        lower = usable["auc"].quantile(criteria.sensitive_quantile) if n_models else np.nan
        upper = usable["auc"].quantile(criteria.resistant_quantile) if n_models else np.nan
        sensitive = int((usable["auc"] <= lower).sum()) if n_models else 0
        resistant = int((usable["auc"] >= upper).sum()) if n_models else 0
        lineages = int(usable["lineage"].dropna().nunique())
        variance = float(usable["auc"].var()) if n_models > 1 else np.nan
        reasons: list[str] = []
        if n_models < criteria.min_models:
            reasons.append("insufficient_models")
        if missing_fraction > criteria.max_missing_fraction:
            reasons.append("excess_missingness")
        if sensitive < criteria.min_sensitive:
            reasons.append("insufficient_sensitive_examples")
        if resistant < criteria.min_resistant:
            reasons.append("insufficient_resistant_examples")
        if lineages < criteria.min_lineages:
            reasons.append("insufficient_lineage_diversity")
        if not np.isfinite(variance) or variance <= 0:
            reasons.append("no_response_variance")
        records.append(
            {
                "drug": drug,
                "n_models": n_models,
                "screened_universe": screened_universe,
                "coverage_fraction": n_models / max(screened_universe, 1),
                "missing_fraction": missing_fraction,
                "response_variance": variance,
                "q25": lower,
                "q75": upper,
                "n_sensitive": sensitive,
                "n_resistant": resistant,
                "n_lineages": lineages,
                "accepted": not reasons,
                "rejection_reason": ";".join(reasons),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["accepted", "response_variance", "n_models"], ascending=[False, False, False]
    )


def select_mvp_drugs(
    audit: pd.DataFrame,
    n_drugs: int = 5,
    preferred: list[str] | None = None,
) -> list[str]:
    """Select accepted drugs, prioritizing understandable oncology mechanisms."""
    accepted = audit[audit["accepted"]].copy()
    if len(accepted) < n_drugs:
        raise ValueError(f"Only {len(accepted)} drugs pass the audit; {n_drugs} are required")
    chosen: list[str] = []
    for drug in preferred or []:
        if drug in set(accepted["drug"]) and drug not in chosen:
            chosen.append(drug)
    for drug in accepted["drug"]:
        if drug not in chosen:
            chosen.append(drug)
        if len(chosen) == n_drugs:
            break
    return chosen[:n_drugs]

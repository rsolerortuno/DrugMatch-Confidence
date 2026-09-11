"""Post-hoc interval feasibility diagnostics, separate from the release policy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def maximum_nominal_coverage(residuals: np.ndarray, maximum_radius: float) -> float:
    """Largest coverage supported by the exact finite-sample residual rank.

    Requires calibration residuals from the same fixed model, not test residuals.
    This is a feasibility diagnostic, not a recommended operating point or a
    conditional classification error guarantee.
    """
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 1 or len(values) < 5 or not np.isfinite(values).all():
        raise ValueError("At least five finite calibration residuals are required")
    if (values < 0).any() or not np.isfinite(maximum_radius):
        raise ValueError("Residuals must be nonnegative and maximum radius finite")
    return float(np.count_nonzero(values <= maximum_radius) / (len(values) + 1))


def candidate_feasibility(frame: pd.DataFrame, margin: float = 0.10) -> pd.DataFrame:
    """Compute headroom to the probability-proposed response tail for each row.

    Positive headroom is the largest symmetric interval radius supporting that
    candidate's proposed class. Negative headroom fails even at radius zero.
    This deliberately excludes release status, feature coverage and OOD gates.
    """
    out = frame.copy()
    out["radius_90"] = (out.interval_upper - out.interval_lower) / 2
    out["class_band_width"] = out.resistant_auc_min - out.sensitive_auc_max
    out["maximum_supportable_radius"] = np.where(
        out.sensitivity_probability >= 0.5,
        out.sensitive_auc_max - out.predicted_auc,
        out.predicted_auc - out.resistant_auc_min,
    )
    out["probability_margin_pass"] = (out.sensitivity_probability - 0.5).abs() >= margin
    out["interval_support_90"] = np.where(
        out.sensitivity_probability >= 0.5,
        out.interval_upper <= out.sensitive_auc_max,
        out.interval_lower >= out.resistant_auc_min,
    )
    out["technical_support_90"] = out.interval_support_90 & out.probability_margin_pass
    out["radius_scale_to_first_support"] = np.divide(
        out.maximum_supportable_radius,
        out.radius_90,
        out=np.full(len(out), np.nan),
        where=out.radius_90.to_numpy() > 0,
    )
    return out


def feasibility_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-fold diagnostics and a radius sensitivity curve, without tuning."""
    candidates = candidate_feasibility(frame)
    rows, curves = [], []
    for (drug, family, fold), group in candidates.groupby(["drug", "family", "fold"]):
        margin_ok = group.loc[group.probability_margin_pass]
        possible = margin_ok.loc[margin_ok.maximum_supportable_radius >= 0]
        rows.append(
            dict(
                drug=drug,
                family=family,
                fold=int(fold),
                n_candidates=len(group),
                radius_90=float(group.radius_90.iloc[0]),
                sensitive_max=float(group.sensitive_auc_max.iloc[0]),
                resistant_min=float(group.resistant_auc_min.iloc[0]),
                class_band_width=float(group.class_band_width.iloc[0]),
                prediction_min=float(group.predicted_auc.min()),
                prediction_max=float(group.predicted_auc.max()),
                maximum_supportable_radius=float(possible.maximum_supportable_radius.max()),
                first_support_radius_scale=float(possible.radius_scale_to_first_support.max()),
                n_interval_support_90=int(group.interval_support_90.sum()),
                n_technical_support_90=int(group.technical_support_90.sum()),
                n_margin_pass=int(group.probability_margin_pass.sum()),
                nominal_coverage_at_first_support=np.nan,
                coverage_status="Unavailable: original fold calibration residuals were not saved",
            )
        )
    for (drug, family), group in candidates.groupby(["drug", "family"]):
        for scale in np.linspace(0, 1, 21):
            eligible = group.probability_margin_pass & (
                group.maximum_supportable_radius >= scale * group.radius_90
            )
            curves.append(
                dict(
                    drug=drug,
                    family=family,
                    radius_scale=float(scale),
                    n_technical_support=int(eligible.sum()),
                    n_candidates=len(group),
                    selection_fraction=float(eligible.mean()),
                    interpretation="Post-hoc radius scaling; not nominal interval coverage or an API policy",
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(curves)


def boundary_tie_summary(group: pd.DataFrame, k: int) -> dict[str, object]:
    """Exact min/max/expected hits over all uniform boundary-tie selections."""
    if not 0 < k <= len(group) or group.predicted_auc.isna().any():
        raise ValueError("Budget must fit the finite ranked candidate set")
    cutoff = group.predicted_auc.nsmallest(k).iloc[-1]
    tied = group[group.predicted_auc.eq(cutoff)]
    before = group[group.predicted_auc.lt(cutoff)]
    slots = k - len(before)
    tied_hits = int((tied.true_auc <= tied.sensitive_auc_max).sum())
    before_hits = int((before.true_auc <= before.sensitive_auc_max).sum())
    return dict(
        hits_tie_min=before_hits + max(0, slots - (len(tied) - tied_hits)),
        hits_tie_max=before_hits + min(slots, tied_hits),
        hits_tie_expected=before_hits + slots * tied_hits / len(tied),
        boundary_tie_size=len(tied),
        boundary_slots=slots,
        tie_policy="Exact saved-score ties; reported list uses ascending ModelID",
    )

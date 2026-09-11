"""Decision-focused, retrospective analysis of saved matched OOF predictions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from drugmatch.policy_audit import (
    boundary_tie_summary,
    feasibility_tables,
    maximum_nominal_coverage,
)
from drugmatch.uncertainty import SplitConformalInterval

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/pierre_fabre_review"


def paired_auc_ci(frame, rng, repetitions=2000):
    tails = frame.dropna(subset=["true_sensitive"])
    wide = tails.pivot(index="model_id", columns="family", values="sensitivity_probability")
    y = (
        tails.drop_duplicates("model_id")
        .set_index("model_id")
        .loc[wide.index, "true_sensitive"]
        .to_numpy()
    )
    strata = [np.flatnonzero(y == v) for v in (0, 1)]
    rows = []
    for baseline in ["elastic_net", "lineage"]:
        diffs = []
        for _ in range(repetitions):
            idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
            diffs.append(
                roc_auc_score(y[idx], wide.xgboost.to_numpy()[idx])
                - roc_auc_score(y[idx], wide[baseline].to_numpy()[idx])
            )
        observed = roc_auc_score(y, wide.xgboost) - roc_auc_score(y, wide[baseline])
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        rows.append(
            dict(
                baseline=baseline,
                auroc_difference=observed,
                ci_lower=lo,
                ci_upper=hi,
                n_extremes=len(y),
            )
        )
    return rows


def budget_table(frame, budgets=(5, 10, 20)):
    rows = []
    for family, group in frame.groupby("family"):
        # Continuous AUC ranking uses all held-out response levels, including the middle.
        ranked = group.sort_values(["predicted_auc", "model_id"], kind="stable")
        prevalence = (group.true_auc <= group.sensitive_auc_max).mean()
        for k in budgets:
            selected = ranked.head(k)
            hits = int((selected.true_auc <= selected.sensitive_auc_max).sum())
            rows.append(
                dict(
                    family=family,
                    k=k,
                    **boundary_tie_summary(group, k),
                    hits=hits,
                    precision=hits / k,
                    random_expected_hits=k * prevalence,
                    enrichment=(hits / k) / prevalence,
                    n_candidates=len(group),
                    mean_observed_auc=float(selected.true_auc.mean()),
                )
            )
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(1701)
    comparisons = []
    risks = []
    budgets = []
    feasibility, sensitivity = [], []
    for path in sorted(OUT.glob("*_matched_oof.csv")):
        frame = pd.read_csv(path)
        drug = frame.drug.iloc[0]
        diagnostics, curve = feasibility_tables(frame)
        calibration_file = OUT / f"{drug}_calibration_residuals.csv"
        if calibration_file.exists():
            residuals = pd.read_csv(calibration_file)
            roles = pd.read_csv(OUT / f"{drug}_fold_roles.csv")
            for index, row in diagnostics.iterrows():
                subset = residuals[residuals.fold.eq(row.fold) & residuals.family.eq(row.family)]
                expected = set(
                    roles.loc[roles.fold.eq(row.fold) & roles.role.eq("calibration"), "model_id"]
                )
                if subset.model_id.duplicated().any() or set(subset.model_id) != expected:
                    raise ValueError("Calibration residual IDs do not match the saved fold roles")
                values = subset.absolute_residual.to_numpy()
                q90 = SplitConformalInterval(0.9).fit(values, np.zeros(len(values))).radius_
                if not np.isclose(q90, row.radius_90, rtol=0, atol=1e-10):
                    raise ValueError("Calibration residuals do not reproduce the saved 90% radius")
                diagnostics.loc[index, "nominal_coverage_at_first_support"] = (
                    maximum_nominal_coverage(values, row.maximum_supportable_radius)
                    if np.isfinite(row.maximum_supportable_radius)
                    else 0.0
                )
                diagnostics.loc[index, "coverage_status"] = (
                    "From matching saved calibration residuals; post-hoc feasibility, not a selected policy"
                )
        feasibility.append(diagnostics)
        sensitivity.append(curve)
        comparisons.extend(dict(drug=drug, **r) for r in paired_auc_ci(frame, rng))
        budgets.append(budget_table(frame).assign(drug=drug))
        for family, group in frame.groupby("family"):
            for margin in [0, 0.1, 0.2, 0.3, 0.4]:
                eligible = group[(group.sensitivity_probability - 0.5).abs() >= margin]
                tails = eligible.dropna(subset=["true_sensitive"])
                risks.append(
                    dict(
                        drug=drug,
                        family=family,
                        probability_margin=margin,
                        coverage=len(eligible) / len(group),
                        n_selected=len(eligible),
                        n_selected_extremes=len(tails),
                        tail_error=float((tails.predicted_sensitive != tails.true_sensitive).mean())
                        if len(tails)
                        else np.nan,
                        intermediate_fraction=float(eligible.true_sensitive.isna().mean())
                        if len(eligible)
                        else np.nan,
                        policy="probability-only diagnostic; does not implement API acceptance",
                    )
                )
        if drug == "trametinib":
            columns = [
                "model_id",
                "lineage",
                "predicted_auc",
                "interval_lower",
                "interval_upper",
                "sensitivity_probability",
                "sensitive_auc_max",
                "resistant_auc_min",
                "technical_eligibility",
                "true_auc",
            ]
            for family in ["xgboost", "elastic_net", "lineage"]:
                frame[frame.family.eq(family)].sort_values(["predicted_auc", "model_id"]).head(10)[
                    columns
                ].to_csv(OUT / f"mapk_top10_{family}.csv", index=False)
    pd.concat(feasibility).to_csv(OUT / "interval_feasibility.csv", index=False)
    pd.concat(sensitivity).to_csv(OUT / "interval_radius_sensitivity.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "paired_auroc_differences.csv", index=False)
    pd.DataFrame(risks).to_csv(OUT / "risk_coverage_diagnostic.csv", index=False)
    pd.concat(budgets).to_csv(OUT / "screening_budget_summary.csv", index=False)
    (OUT / "analysis_protocol.json").write_text(
        json.dumps(
            dict(
                seed=1701,
                split_stratification="Outer and inner fold allocation uses full-cohort outcome-rank quartiles. Outcomes inform allocation only, not fitted features, class cutoffs or calibration. This is outcome-stratified retrospective evaluation, not outcome-blind prospective assignment.",
                tie_analysis="Exact saved-score boundary ties: deterministic ModelID list, min/max hits, and analytic expectation over uniform boundary tie breaks. Outcomes evaluate, never select, a tie break.",
                interval_feasibility="Per-fold radius and candidate headroom; post-hoc scaling is not nominal coverage. Original calibration residuals are unavailable, so 65–79% first-support coverage estimates are not verified. Residual error is not an assay-noise ceiling.",
                bootstrap_repetitions=2000,
                confidence_interval="Percentile paired stratified bootstrap of saved OOF scores; conditional on fitted folds, not retraining uncertainty.",
                ranking="Ascending predicted assay AUC; deterministic ModelID tie break; all response levels included.",
                hit_definition="Observed assay AUC at or below the fit-only 25th percentile in the corresponding fold.",
                data_status="Retrospective reanalysis, not a prospective validation; no tuning on external outcomes.",
                risk_coverage="Probability-only diagnostic reported separately from the frozen full abstention policy.",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

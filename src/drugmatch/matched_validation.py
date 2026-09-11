"""Matched out-of-fold evaluation of fixed XGBoost, linear and lineage baselines.

Every family sees identical fit/calibration/test IDs and fit-only feature/label
selection. This estimates a fixed development protocol, not the tuned release.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from drugmatch.calibration import ProbabilityCalibrator
from drugmatch.cross_validation import DEFAULT_REGRESSION_PARAMS
from drugmatch.evaluation import classification_metrics, regression_metrics
from drugmatch.models import train_baselines, train_xgboost
from drugmatch.preprocessing import select_training_features
from drugmatch.training import _assign_classes
from drugmatch.uncertainty import SplitConformalInterval


def matched_oof(
    features,
    response,
    metadata,
    drug,
    n_splits=5,
    seed=2026,
    model_params=None,
    calibration_output=None,
):
    """Return long-form held-out predictions and explicit split-role membership."""
    selected = (
        response.loc[response.drug.str.lower().eq(drug.lower())]
        .dropna(subset=["auc", "model_id"])
        .drop_duplicates("model_id")
        .set_index("model_id")
    )
    ids = features.index.intersection(selected.index).intersection(metadata.index)
    auc = selected.loc[ids, "auc"].astype(float)
    if not ids.is_unique or len(ids) < 80 or not np.isfinite(auc).all():
        raise ValueError("Require at least 80 unique models with finite response")
    # Full-cohort outcomes inform stratified allocation only; fitting remains fold-local.
    strata = pd.qcut(auc.rank(method="first"), 4, labels=False)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows, memberships = [], []
    params = dict(DEFAULT_REGRESSION_PARAMS)
    params.update(model_params or {})
    for fold, (dev_pos, test_pos) in enumerate(folds.split(ids, strata), 1):
        dev, test = ids[dev_pos], ids[test_pos]
        fit, calibration = train_test_split(
            dev.to_numpy(dtype=object),
            test_size=0.20,
            random_state=seed + fold,
            stratify=strata.loc[dev],
        )
        fit, calibration = pd.Index(fit), pd.Index(calibration)
        for role, members in [("fit", fit), ("calibration", calibration), ("test", test)]:
            memberships.extend(
                {"drug": drug, "fold": fold, "role": role, "model_id": m} for m in members
            )
        lower, upper = auc.loc[fit].quantile([0.25, 0.75])
        labels = _assign_classes(auc, lower, upper).binary_class
        fit_cls, cal_cls = (
            fit.intersection(labels.dropna().index),
            calibration.intersection(labels.dropna().index),
        )
        if min(labels.loc[fit_cls].nunique(), labels.loc[cal_cls].nunique()) < 2:
            raise ValueError(f"{drug} fold {fold}: one response class missing")
        columns, _ = select_training_features(features.loc[fit], drug)
        X = features.loc[ids, columns]
        regs, clss = train_baselines(
            X.loc[fit], auc.loc[fit], labels.loc[fit_cls].astype(int), seed + fold
        )
        reg, cls = train_xgboost(
            X.loc[fit], auc.loc[fit], labels.loc[fit_cls].astype(int), params, params, seed + fold
        )
        regs["xgboost"], clss["xgboost"] = reg, cls
        for family in ("lineage", "elastic_net", "xgboost"):
            if family not in regs:
                continue
            cols = ["meta::lineage"] if family == "lineage" else columns
            reg, cls = regs[family], clss[family]
            pred = reg.predict(X.loc[test, cols])
            interval = SplitConformalInterval(0.90).fit(
                auc.loc[calibration].to_numpy(), reg.predict(X.loc[calibration, cols])
            )
            if calibration_output is not None:
                calibration_output.extend(
                    dict(
                        drug=drug,
                        family=family,
                        fold=fold,
                        model_id=model_id,
                        absolute_residual=float(residual),
                    )
                    for model_id, residual in zip(
                        calibration, interval.calibration_residuals_, strict=True
                    )
                )
            lo, hi = interval.predict(pred)
            calibrator = ProbabilityCalibrator("platt").fit(
                cls.predict_proba(X.loc[cal_cls, cols])[:, 1],
                labels.loc[cal_cls].astype(int).to_numpy(),
            )
            probability = calibrator.predict(cls.predict_proba(X.loc[test, cols])[:, 1])
            for i, model_id in enumerate(test):
                sensitive = probability[i] >= 0.5
                supported = hi[i] <= lower if sensitive else lo[i] >= upper
                # Technical eligibility only: does not waive model evidence/OOD checks in the API.
                eligible = bool(supported and abs(probability[i] - 0.5) >= 0.10)
                rows.append(
                    dict(
                        drug=drug,
                        family=family,
                        fold=fold,
                        model_id=model_id,
                        lineage=metadata.loc[model_id, "lineage"],
                        true_auc=auc.loc[model_id],
                        predicted_auc=float(pred[i]),
                        interval_lower=float(lo[i]),
                        interval_upper=float(hi[i]),
                        true_sensitive=labels.loc[model_id],
                        sensitivity_probability=float(probability[i]),
                        predicted_sensitive=int(sensitive),
                        decision_threshold=0.5,
                        sensitive_auc_max=float(lower),
                        resistant_auc_min=float(upper),
                        technical_eligibility=eligible,
                        n_fit=len(fit),
                        n_calibration=len(calibration),
                    )
                )
    return pd.DataFrame(rows), pd.DataFrame(memberships)


def summarize_matched(predictions):
    """Pool only OOF predictions; report class-tail support and all-response error."""
    rows = []
    for (drug, family), group in predictions.groupby(["drug", "family"]):
        tails = group.dropna(subset=["true_sensitive"])
        cls = classification_metrics(
            tails.true_sensitive.to_numpy(), tails.sensitivity_probability.to_numpy()
        )
        reg = regression_metrics(group.true_auc.to_numpy(), group.predicted_auc.to_numpy())
        eligible = group[group.technical_eligibility]
        accepted_tails = eligible.dropna(subset=["true_sensitive"])
        rows.append(
            dict(
                drug=drug,
                family=family,
                n_models=len(group),
                n_extremes=len(tails),
                **{f"classification_{k}": v for k, v in cls.items()},
                **{f"regression_{k}": v for k, v in reg.items()},
                interval_coverage=float(
                    (
                        (group.true_auc >= group.interval_lower)
                        & (group.true_auc <= group.interval_upper)
                    ).mean()
                ),
                technical_coverage=len(eligible) / len(group),
                n_eligible=len(eligible),
                n_eligible_extremes=len(accepted_tails),
                eligible_tail_error=float(
                    (accepted_tails.predicted_sensitive != accepted_tails.true_sensitive).mean()
                )
                if len(accepted_tails)
                else np.nan,
                eligible_intermediate_fraction=float(eligible.true_sensitive.isna().mean())
                if len(eligible)
                else np.nan,
            )
        )
    return pd.DataFrame(rows)

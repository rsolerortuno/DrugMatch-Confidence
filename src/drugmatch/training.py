"""End-to-end training for a single drug with confidence components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from drugmatch.calibration import ProbabilityCalibrator
from drugmatch.evaluation import classification_metrics, regression_metrics
from drugmatch.explain import global_importance
from drugmatch.models import (
    train_baselines,
    train_xgboost_classifier,
    train_xgboost_regressor,
)
from drugmatch.ood import PCADistanceOOD
from drugmatch.preprocessing import select_training_features
from drugmatch.splitting import grouped_split
from drugmatch.uncertainty import SplitConformalInterval
from drugmatch.utils import stable_json_hash, write_json

DISCLAIMER = (
    "For preclinical research use only. Predictions concern cancer cell-line models and must not "
    "be interpreted as patient treatment recommendations."
)


@dataclass
class TrainingOutcome:
    drug: str
    n_models: int
    n_train: int
    n_validation: int
    n_test: int
    regression_metrics: dict[str, float]
    classification_metrics: dict[str, float]
    baseline_metrics: dict[str, dict[str, float]]
    interval_coverage: float
    model_path: str


def _index_members(index: pd.Index, members: list[str]) -> pd.Index:
    return index.intersection(pd.Index(members))


def _assign_classes(auc: pd.Series, lower: float, upper: float) -> pd.DataFrame:
    frame = pd.DataFrame({"auc": auc.astype(float)})
    frame["response_percentile"] = frame["auc"].rank(pct=True, method="average")
    frame["class"] = "intermediate"
    frame.loc[frame["auc"] <= lower, "class"] = "sensitive"
    frame.loc[frame["auc"] >= upper, "class"] = "resistant"
    frame["binary_class"] = frame["class"].map({"sensitive": 1, "resistant": 0})
    return frame


def _regression_candidates() -> list[dict[str, Any]]:
    return [
        {},
        {"max_depth": 2, "min_child_weight": 5, "n_estimators": 400, "learning_rate": 0.03},
        {"max_depth": 4, "min_child_weight": 3, "n_estimators": 260, "learning_rate": 0.04},
        {
            "max_depth": 3,
            "min_child_weight": 7,
            "n_estimators": 350,
            "learning_rate": 0.03,
            "reg_lambda": 2.0,
            "colsample_bytree": 0.65,
        },
    ]


def _classification_candidates() -> list[dict[str, Any]]:
    return [
        {},
        {"max_depth": 2, "min_child_weight": 4, "n_estimators": 400, "learning_rate": 0.03},
        {"max_depth": 4, "min_child_weight": 3, "n_estimators": 260, "learning_rate": 0.04},
        {
            "max_depth": 3,
            "min_child_weight": 7,
            "n_estimators": 350,
            "learning_rate": 0.03,
            "reg_lambda": 2.0,
            "colsample_bytree": 0.65,
        },
    ]


def _select_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    seed: int,
    tune: bool,
    override: dict[str, Any] | None,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    candidates = _regression_candidates() if tune else [{}]
    rows: list[dict[str, Any]] = []
    fitted: list[Pipeline] = []
    for index, candidate in enumerate(candidates):
        params = dict(candidate)
        params.update(override or {})
        model = train_xgboost_regressor(X_train, y_train, params=params, seed=seed + index)
        metrics = regression_metrics(y_validation.to_numpy(), model.predict(X_validation))
        rows.append({"candidate": index, "params": params, **metrics})
        fitted.append(model)
    table = pd.DataFrame(rows)
    ranking = table.sort_values(["spearman", "rmse"], ascending=[False, True])
    best_index = int(ranking.iloc[0]["candidate"])
    return fitted[best_index], dict(rows[best_index]["params"]), table


def _select_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    seed: int,
    tune: bool,
    override: dict[str, Any] | None,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    candidates = _classification_candidates() if tune else [{}]
    rows: list[dict[str, Any]] = []
    fitted: list[Pipeline] = []
    for index, candidate in enumerate(candidates):
        params = dict(candidate)
        params.update(override or {})
        model = train_xgboost_classifier(X_train, y_train, params=params, seed=seed + index)
        probabilities = model.predict_proba(X_validation)[:, 1]
        metrics = classification_metrics(y_validation.to_numpy(), probabilities)
        rows.append({"candidate": index, "params": params, **metrics})
        fitted.append(model)
    table = pd.DataFrame(rows)
    ranking = table.sort_values(["auprc", "auroc", "brier"], ascending=[False, False, True])
    best_index = int(ranking.iloc[0]["candidate"])
    return fitted[best_index], dict(rows[best_index]["params"]), table


def train_drug_bundle(
    drug: str,
    features: pd.DataFrame,
    response: pd.DataFrame,
    metadata: pd.DataFrame,
    output_directory: str | Path,
    data_release: str,
    seed: int = 42,
    regression_params: dict[str, Any] | None = None,
    classification_params: dict[str, Any] | None = None,
    tune: bool = True,
) -> TrainingOutcome:
    """Train, tune, calibrate, evaluate and serialize all artifacts for one drug."""
    selected_response = response[response["drug"].astype(str).str.lower().eq(drug.lower())].copy()
    selected_response = selected_response.dropna(subset=["model_id", "auc"]).drop_duplicates(
        "model_id"
    )
    selected_response["model_id"] = selected_response["model_id"].astype(str)
    selected_response = selected_response.set_index("model_id")
    common = features.index.intersection(selected_response.index).intersection(metadata.index)
    if len(common) < 80:
        raise ValueError(
            f"Drug {drug!r} has only {len(common)} models with complete requested modalities"
        )
    X_full = features.loc[common]
    auc = selected_response.loc[common, "auc"].astype(float)
    meta = metadata.loc[common].copy()

    response_strata = pd.qcut(auc.rank(method="first"), q=4, labels=["q1", "q2", "q3", "q4"])
    split = grouped_split(common, seed=seed, stratify=response_strata)
    train_ids = _index_members(common, split.train_ids)
    validation_ids = _index_members(common, split.validation_ids)
    test_ids = _index_members(common, split.test_ids)

    # Preserve the historic test and calibration memberships. Tuning uses only
    # a new inner partition of the original training set.
    fit_members, tuning_members = train_test_split(
        train_ids.to_numpy(dtype=object),
        test_size=0.20,
        random_state=seed + 2,
        stratify=response_strata.loc[train_ids],
    )
    train_ids = _index_members(common, list(fit_members))
    tuning_ids = _index_members(common, list(tuning_members))
    lower = float(auc.loc[train_ids].quantile(0.25))
    upper = float(auc.loc[train_ids].quantile(0.75))
    y = _assign_classes(auc, lower, upper)
    train_cls_ids = y.loc[train_ids].dropna(subset=["binary_class"]).index
    validation_cls_ids = y.loc[validation_ids].dropna(subset=["binary_class"]).index
    tuning_cls_ids = y.loc[tuning_ids].dropna(subset=["binary_class"]).index
    test_cls_ids = y.loc[test_ids].dropna(subset=["binary_class"]).index
    for partition, ids in {
        "train": train_cls_ids,
        "tuning": tuning_cls_ids,
        "validation": validation_cls_ids,
        "test": test_cls_ids,
    }.items():
        if len(ids) < 4 or y.loc[ids, "binary_class"].nunique() < 2:
            raise ValueError(
                f"{drug}: {partition} split does not contain enough sensitive and resistant models"
            )

    selected_columns, feature_manifest = select_training_features(X_full.loc[train_ids], drug)
    X = X_full[selected_columns].copy()
    X_train, X_validation, X_test = X.loc[train_ids], X.loc[validation_ids], X.loc[test_ids]
    y_train_reg = y.loc[train_ids, "auc"]
    y_validation_reg = y.loc[validation_ids, "auc"]
    y_test_reg = y.loc[test_ids, "auc"]
    y_train_cls = y.loc[train_cls_ids, "binary_class"].astype(int)
    y_validation_cls = y.loc[validation_cls_ids, "binary_class"].astype(int)
    y_test_cls = y.loc[test_cls_ids, "binary_class"].astype(int)

    regressors, classifiers = train_baselines(X_train, y_train_reg, y_train_cls, seed=seed)
    regression, selected_regression_params, regression_tuning = _select_regressor(
        X_train,
        y_train_reg,
        X.loc[tuning_ids],
        y.loc[tuning_ids, "auc"],
        seed,
        tune,
        regression_params,
    )
    classification, selected_classification_params, classification_tuning = _select_classifier(
        X.loc[train_cls_ids],
        y_train_cls,
        X.loc[tuning_cls_ids],
        y.loc[tuning_cls_ids, "binary_class"].astype(int),
        seed,
        tune,
        classification_params,
    )

    validation_reg_pred = regression.predict(X_validation)
    conformal = SplitConformalInterval(coverage=0.90).fit(
        y_validation_reg.to_numpy(), validation_reg_pred
    )
    test_reg_pred = regression.predict(X_test)
    test_lower, test_upper = conformal.predict(test_reg_pred)
    regression_result = regression_metrics(y_test_reg.to_numpy(), test_reg_pred)
    coverage = conformal.empirical_coverage(y_test_reg.to_numpy(), test_reg_pred)

    raw_validation_probs = classification.predict_proba(X.loc[validation_cls_ids])[:, 1]
    # Predeclared calibration family and threshold: no selection on calibration data.
    calibrator = ProbabilityCalibrator(method="platt").fit(
        raw_validation_probs, y_validation_cls.to_numpy()
    )
    calibrated_validation_probs = calibrator.predict(raw_validation_probs)
    decision_threshold = 0.5
    raw_test_probs = classification.predict_proba(X.loc[test_cls_ids])[:, 1]
    calibrated_test_probs = calibrator.predict(raw_test_probs)
    classification_result = classification_metrics(
        y_test_cls.to_numpy(), calibrated_test_probs, threshold=decision_threshold
    )

    transformed_train = regression.named_steps["preprocess"].transform(X_train)
    ood = PCADistanceOOD(n_components=min(8, transformed_train.shape[1], len(train_ids) - 1)).fit(
        transformed_train
    )

    baseline_results: dict[str, dict[str, float]] = {}
    mean_prediction = regressors["mean"].predict(np.zeros((len(X_test), 1)))
    baseline_results["mean_regression"] = regression_metrics(y_test_reg.to_numpy(), mean_prediction)
    baseline_results["elastic_net_regression"] = regression_metrics(
        y_test_reg.to_numpy(), regressors["elastic_net"].predict(X_test)
    )
    if "lineage" in regressors:
        baseline_results["lineage_regression"] = regression_metrics(
            y_test_reg.to_numpy(), regressors["lineage"].predict(X_test[["meta::lineage"]])
        )
    majority_probs = classifiers["majority"].predict_proba(np.zeros((len(test_cls_ids), 1)))
    positive_column = list(classifiers["majority"].classes_).index(1)
    baseline_results["majority_classification"] = classification_metrics(
        y_test_cls.to_numpy(), majority_probs[:, positive_column]
    )
    baseline_results["elastic_net_classification"] = classification_metrics(
        y_test_cls.to_numpy(), classifiers["elastic_net"].predict_proba(X.loc[test_cls_ids])[:, 1]
    )
    if "lineage" in classifiers:
        baseline_results["lineage_classification"] = classification_metrics(
            y_test_cls.to_numpy(),
            classifiers["lineage"].predict_proba(X.loc[test_cls_ids, ["meta::lineage"]])[:, 1],
        )

    importance = global_importance(classification, X.loc[test_cls_ids])
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stem = drug.lower().replace(" ", "_")
    bundle_path = output / f"{stem}.joblib"

    regression_predictions = pd.DataFrame(
        {
            "model_id": test_ids,
            "true_auc": y_test_reg.loc[test_ids].to_numpy(),
            "predicted_auc": test_reg_pred,
            "interval_lower": test_lower,
            "interval_upper": test_upper,
            "lineage": meta.loc[test_ids, "lineage"].to_numpy(),
        }
    )
    classification_predictions = pd.DataFrame(
        {
            "model_id": test_cls_ids,
            "true_sensitive": y_test_cls.loc[test_cls_ids].to_numpy(),
            "raw_probability": raw_test_probs,
            "sensitivity_probability": calibrated_test_probs,
            "predicted_sensitive": (calibrated_test_probs >= decision_threshold).astype(int),
            "lineage": meta.loc[test_cls_ids, "lineage"].to_numpy(),
        }
    )
    validation_predictions = pd.DataFrame(
        {
            "model_id": validation_cls_ids,
            "true_sensitive": y_validation_cls.loc[validation_cls_ids].to_numpy(),
            "raw_probability": raw_validation_probs,
            "sensitivity_probability": calibrated_validation_probs,
        }
    )

    bundle = {
        "drug": drug,
        "regression": regression,
        "classification": classification,
        "calibrator": calibrator,
        "conformal": conformal,
        "ood": ood,
        "feature_columns": list(X.columns),
        "feature_manifest": feature_manifest.to_dict(orient="records"),
        "training_reference": X_train,
        "data_release": data_release,
        "project_version": "1.1.0",
        "validation_protocol": "independent_calibration_v2",
        "validation_status": "unreviewed",
        "partition_roles": {
            "fit": list(train_ids),
            "tuning": list(tuning_ids),
            "calibration": list(validation_ids),
            "test": list(test_ids),
        },
        "disclaimer": DISCLAIMER,
        "split": split.model_dump(),
        "class_definition": {"positive": "sensitive", "negative": "resistant"},
        "class_thresholds": {"sensitive_auc_max": lower, "resistant_auc_min": upper},
        # Assay AUC is not assumed to have a hard [0, 1] support.
        "response_bounds": None,
        "decision_threshold": decision_threshold,
        "calibration_method": calibrator.method,
        "selected_regression_params": selected_regression_params,
        "selected_classification_params": selected_classification_params,
        "metrics": {
            "regression": regression_result,
            "classification": classification_result,
            "baselines": baseline_results,
            "interval_coverage": coverage,
            "decision_threshold": decision_threshold,
            "calibration_method": calibrator.method,
            "n_models": len(common),
            "n_selected_features": len(selected_columns),
        },
        "artifact_hash": stable_json_hash(
            {
                "drug": drug,
                "columns": list(X.columns),
                "release": data_release,
                "seed": seed,
                "thresholds": [lower, upper],
            }
        ),
    }
    joblib.dump(bundle, bundle_path, compress=3)
    importance.to_csv(output / f"{stem}_shap.csv", index=False)
    feature_manifest.to_csv(output / f"{stem}_feature_manifest.csv", index=False)
    regression_tuning.assign(params=regression_tuning["params"].astype(str)).to_csv(
        output / f"{stem}_regression_tuning.csv", index=False
    )
    classification_tuning.assign(params=classification_tuning["params"].astype(str)).to_csv(
        output / f"{stem}_classification_tuning.csv", index=False
    )
    regression_predictions.to_csv(
        output / f"{stem}_internal_regression_predictions.csv", index=False
    )
    classification_predictions.to_csv(
        output / f"{stem}_internal_classification_predictions.csv", index=False
    )
    validation_predictions.to_csv(output / f"{stem}_validation_predictions.csv", index=False)
    write_json(output / f"{stem}_metrics.json", bundle["metrics"])
    write_json(output / f"{stem}_split.json", split.model_dump())
    write_json(output / f"{stem}_partition_roles.json", bundle["partition_roles"])

    return TrainingOutcome(
        drug=drug,
        n_models=len(common),
        n_train=len(train_ids),
        n_validation=len(validation_ids),
        n_test=len(test_ids),
        regression_metrics=regression_result,
        classification_metrics=classification_result,
        baseline_metrics=baseline_results,
        interval_coverage=coverage,
        model_path=str(bundle_path),
    )

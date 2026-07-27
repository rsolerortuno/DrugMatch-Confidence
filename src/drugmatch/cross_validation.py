"""Five-fold out-of-fold validation for stable portfolio performance estimates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from drugmatch.calibration import ProbabilityCalibrator
from drugmatch.evaluation import classification_metrics, regression_metrics
from drugmatch.models import train_xgboost_classifier, train_xgboost_regressor
from drugmatch.preprocessing import select_training_features

DEFAULT_REGRESSION_PARAMS: dict[str, Any] = {
    "n_estimators": 250,
    "learning_rate": 0.03,
    "max_depth": 2,
    "min_child_weight": 5,
    "reg_lambda": 2.0,
    "colsample_bytree": 0.75,
    "n_jobs": 1,
}
DEFAULT_CLASSIFICATION_PARAMS: dict[str, Any] = dict(DEFAULT_REGRESSION_PARAMS)


def cross_validated_predictions(
    features: pd.DataFrame,
    response: pd.DataFrame,
    metadata: pd.DataFrame,
    drug: str,
    n_splits: int = 5,
    seed: int = 2026,
) -> tuple[dict[str, float], dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Return leakage-safe OOF regression and calibrated classification predictions."""
    selected = response[response["drug"].astype(str).str.lower().eq(drug.lower())].copy()
    selected = (
        selected.dropna(subset=["model_id", "auc"])
        .drop_duplicates("model_id")
        .set_index("model_id")
    )
    common = features.index.intersection(selected.index).intersection(metadata.index)
    X_full = features.loc[common]
    auc = selected.loc[common, "auc"].astype(float)
    strata = pd.qcut(auc.rank(method="first"), q=4, labels=False)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    regression_rows: list[dict[str, object]] = []
    classification_rows: list[dict[str, object]] = []

    for fold, (train_pos, test_pos) in enumerate(splitter.split(common, strata), start=1):
        train_ids = common[train_pos]
        test_ids = common[test_pos]
        lower = float(auc.loc[train_ids].quantile(0.25))
        upper = float(auc.loc[train_ids].quantile(0.75))
        selected_columns, _ = select_training_features(X_full.loc[train_ids], drug)
        X = X_full[selected_columns]

        regressor = train_xgboost_regressor(
            X.loc[train_ids], auc.loc[train_ids], params=DEFAULT_REGRESSION_PARAMS, seed=seed + fold
        )
        reg_prediction = regressor.predict(X.loc[test_ids])
        for model_id, true_value, predicted_value in zip(
            test_ids, auc.loc[test_ids].to_numpy(), reg_prediction, strict=True
        ):
            regression_rows.append(
                {
                    "model_id": model_id,
                    "fold": fold,
                    "true_auc": float(true_value),
                    "predicted_auc": float(predicted_value),
                    "lineage": metadata.loc[model_id, "lineage"],
                }
            )

        class_labels = pd.Series(np.nan, index=common, dtype=float)
        class_labels.loc[auc <= lower] = 1.0
        class_labels.loc[auc >= upper] = 0.0
        train_cls_ids = train_ids.intersection(class_labels.dropna().index)
        test_cls_ids = test_ids.intersection(class_labels.dropna().index)
        y_train = class_labels.loc[train_cls_ids].astype(int)
        if len(test_cls_ids) < 5 or y_train.nunique() < 2:
            continue
        model_ids, calibration_ids = train_test_split(
            train_cls_ids,
            test_size=0.20,
            random_state=seed + fold,
            stratify=y_train.loc[train_cls_ids],
        )
        classifier = train_xgboost_classifier(
            X.loc[model_ids],
            y_train.loc[model_ids],
            params=DEFAULT_CLASSIFICATION_PARAMS,
            seed=seed + fold,
        )
        calibration_raw = classifier.predict_proba(X.loc[calibration_ids])[:, 1]
        calibrator = ProbabilityCalibrator(method="platt").fit(
            calibration_raw, y_train.loc[calibration_ids].to_numpy()
        )
        calibration_probability = calibrator.predict(calibration_raw)
        thresholds = np.linspace(0.05, 0.95, 91)
        threshold = float(
            max(
                thresholds,
                key=lambda value: classification_metrics(
                    y_train.loc[calibration_ids].to_numpy(),
                    calibration_probability,
                    threshold=value,
                )["balanced_accuracy"],
            )
        )
        test_probability = calibrator.predict(classifier.predict_proba(X.loc[test_cls_ids])[:, 1])
        for model_id, probability in zip(test_cls_ids, test_probability, strict=True):
            classification_rows.append(
                {
                    "model_id": model_id,
                    "fold": fold,
                    "true_sensitive": int(class_labels.loc[model_id]),
                    "sensitivity_probability": float(probability),
                    "fold_decision_threshold": threshold,
                    "predicted_sensitive": int(probability >= threshold),
                    "lineage": metadata.loc[model_id, "lineage"],
                }
            )

    regression_predictions = pd.DataFrame(regression_rows)
    classification_predictions = pd.DataFrame(classification_rows)
    reg_metrics = regression_metrics(
        regression_predictions["true_auc"].to_numpy(),
        regression_predictions["predicted_auc"].to_numpy(),
    )
    cls_metrics = classification_metrics(
        classification_predictions["true_sensitive"].to_numpy(),
        classification_predictions["sensitivity_probability"].to_numpy(),
        threshold=0.5,
    )
    cls_metrics["fold_specific_threshold_accuracy"] = float(
        (
            classification_predictions["true_sensitive"]
            == classification_predictions["predicted_sensitive"]
        ).mean()
    )
    return reg_metrics, cls_metrics, regression_predictions, classification_predictions

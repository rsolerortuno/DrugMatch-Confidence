"""Baseline, Elastic Net and XGBoost training primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from drugmatch.preprocessing import build_preprocessor


def _elastic_net_logistic_kwargs() -> dict[str, Any]:
    major, minor = (int(part) for part in sklearn_version.split(".")[:2])
    if (major, minor) >= (1, 8):
        return {"l1_ratio": 0.5}
    return {"penalty": "elasticnet", "l1_ratio": 0.5}


@dataclass
class FittedModels:
    regression: Pipeline
    classification: Pipeline
    regression_baselines: dict[str, BaseEstimator]
    classification_baselines: dict[str, BaseEstimator]


def train_baselines(
    X_train: pd.DataFrame,
    y_regression: pd.Series,
    y_classification: pd.Series,
    seed: int = 42,
) -> tuple[dict[str, BaseEstimator], dict[str, BaseEstimator]]:
    """Train transparent naive, lineage-only and Elastic Net baselines."""
    preprocessor = build_preprocessor(X_train)
    regressors: dict[str, BaseEstimator] = {
        "mean": DummyRegressor(strategy="mean").fit(np.zeros((len(y_regression), 1)), y_regression),
        "elastic_net": Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", ElasticNet(alpha=0.02, l1_ratio=0.5, max_iter=5000, random_state=seed)),
            ]
        ).fit(X_train, y_regression),
    }
    lineage_columns = [column for column in X_train if column == "meta::lineage"]
    if lineage_columns:
        regressors["lineage"] = Pipeline(
            [
                ("preprocess", build_preprocessor(X_train[lineage_columns])),
                ("model", Ridge(alpha=1.0)),
            ]
        ).fit(X_train[lineage_columns], y_regression)
    classifier_preprocessor = build_preprocessor(X_train.loc[y_classification.index])
    classifiers: dict[str, BaseEstimator] = {
        "majority": DummyClassifier(strategy="prior").fit(
            np.zeros((len(y_classification), 1)), y_classification
        ),
        "elastic_net": Pipeline(
            [
                ("preprocess", classifier_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        C=1.0,
                        **_elastic_net_logistic_kwargs(),
                        max_iter=1500,
                        tol=1e-3,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ).fit(X_train.loc[y_classification.index], y_classification),
    }
    if lineage_columns:
        lineage_classification_X = X_train.loc[y_classification.index, lineage_columns]
        classifiers["lineage"] = Pipeline(
            [
                ("preprocess", build_preprocessor(lineage_classification_X)),
                (
                    "model",
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ).fit(lineage_classification_X, y_classification)
    return regressors, classifiers


def _regression_defaults(seed: int) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "n_estimators": 300,
        "learning_rate": 0.04,
        "max_depth": 3,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "n_jobs": 4,
        "random_state": seed,
    }


def _classification_defaults(seed: int) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 300,
        "learning_rate": 0.04,
        "max_depth": 3,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "n_jobs": 4,
        "random_state": seed,
    }


def train_xgboost_regressor(
    X_train: pd.DataFrame,
    y_regression: pd.Series,
    params: dict[str, Any] | None = None,
    seed: int = 42,
) -> Pipeline:
    defaults = _regression_defaults(seed)
    defaults.update(params or {})
    model = Pipeline(
        [
            ("preprocess", build_preprocessor(X_train)),
            ("model", XGBRegressor(**defaults)),
        ]
    )
    return model.fit(X_train, y_regression)


def train_xgboost_classifier(
    X_train: pd.DataFrame,
    y_classification: pd.Series,
    params: dict[str, Any] | None = None,
    seed: int = 42,
) -> Pipeline:
    defaults = _classification_defaults(seed)
    defaults.update(params or {})
    model = Pipeline(
        [
            ("preprocess", build_preprocessor(X_train)),
            ("model", XGBClassifier(**defaults)),
        ]
    )
    return model.fit(X_train, y_classification)


def train_xgboost(
    X_train: pd.DataFrame,
    y_regression: pd.Series,
    y_classification: pd.Series,
    regression_params: dict[str, Any] | None = None,
    classification_params: dict[str, Any] | None = None,
    seed: int = 42,
) -> tuple[Pipeline, Pipeline]:
    """Train one regression and one binary classification pipeline."""
    regression = train_xgboost_regressor(X_train, y_regression, regression_params, seed)
    classification_X = X_train.loc[y_classification.index]
    classification = train_xgboost_classifier(
        classification_X, y_classification, classification_params, seed
    )
    return regression, classification

"""Standalone evaluation of naive, lineage and Elastic Net baselines."""

from __future__ import annotations

import pandas as pd

from drugmatch.evaluation import classification_metrics, regression_metrics
from drugmatch.labels import labels_for_drug
from drugmatch.models import train_baselines
from drugmatch.splitting import grouped_split


def evaluate_baselines(
    features: pd.DataFrame,
    response: pd.DataFrame,
    metadata: pd.DataFrame,
    drug: str,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Fit baseline models on train and evaluate once on the fixed internal test split."""
    labels = labels_for_drug(response, drug)
    common = features.index.intersection(labels.index).intersection(metadata.index)
    X = features.loc[common]
    y = labels.loc[common]
    split = grouped_split(common, seed=seed, stratify=metadata.loc[common, "lineage"])
    train_ids = pd.Index(split.train_ids).intersection(common)
    test_ids = pd.Index(split.test_ids).intersection(common)
    train_class_ids = y.loc[train_ids].dropna(subset=["binary_class"]).index
    test_class_ids = y.loc[test_ids].dropna(subset=["binary_class"]).index
    regressors, classifiers = train_baselines(
        X.loc[train_ids],
        y.loc[train_ids, "auc"],
        y.loc[train_class_ids, "binary_class"].astype(int),
        seed=seed,
    )
    results: dict[str, dict[str, float]] = {
        "mean_regression": regression_metrics(
            y.loc[test_ids, "auc"], regressors["mean"].predict([[0]] * len(test_ids))
        ),
        "elastic_net_regression": regression_metrics(
            y.loc[test_ids, "auc"], regressors["elastic_net"].predict(X.loc[test_ids])
        ),
        "majority_classification": classification_metrics(
            y.loc[test_class_ids, "binary_class"].astype(int),
            classifiers["majority"].predict_proba([[0]] * len(test_class_ids))[
                :, list(classifiers["majority"].classes_).index(1)
            ],
        ),
        "elastic_net_classification": classification_metrics(
            y.loc[test_class_ids, "binary_class"].astype(int),
            classifiers["elastic_net"].predict_proba(X.loc[test_class_ids])[:, 1],
        ),
    }
    if "lineage" in regressors:
        results["lineage_regression"] = regression_metrics(
            y.loc[test_ids, "auc"],
            regressors["lineage"].predict(X.loc[test_ids, ["meta::lineage"]]),
        )
    if "lineage" in classifiers:
        results["lineage_classification"] = classification_metrics(
            y.loc[test_class_ids, "binary_class"].astype(int),
            classifiers["lineage"].predict_proba(X.loc[test_class_ids, ["meta::lineage"]])[:, 1],
        )
    return results

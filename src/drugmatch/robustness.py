"""Robustness analyses: lineage transfer, modality ablations and SHAP stability."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from drugmatch.evaluation import classification_metrics, regression_metrics
from drugmatch.explain import global_importance
from drugmatch.labels import labels_for_drug
from drugmatch.models import train_xgboost, train_xgboost_classifier
from drugmatch.ood import PCADistanceOOD
from drugmatch.preprocessing import select_training_features
from drugmatch.splitting import grouped_split


def _columns_for_modalities(features: pd.DataFrame, modalities: set[str]) -> list[str]:
    prefixes = {
        "expression": "expr::",
        "mutations": "mut::",
        "copy_number": "cn::",
        "signatures": "sig::",
        "lineage": "meta::lineage",
    }
    columns: list[str] = []
    for modality in modalities:
        prefix = prefixes[modality]
        if modality == "lineage":
            columns.extend([column for column in features if column == prefix])
        else:
            columns.extend([column for column in features if column.startswith(prefix)])
    return columns


def modality_ablation(
    features: pd.DataFrame,
    response: pd.DataFrame,
    metadata: pd.DataFrame,
    drug: str,
    seed: int = 42,
    model_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Evaluate fixed feature-group combinations on one immutable grouped split."""
    labels = labels_for_drug(response, drug)
    common = features.index.intersection(labels.index).intersection(metadata.index)
    X = features.loc[common]
    y = labels.loc[common]
    split = grouped_split(common, seed=seed, stratify=y["response_percentile"].mul(4).astype(int))
    train_ids = pd.Index(split.train_ids).intersection(common)
    test_ids = pd.Index(split.test_ids).intersection(common)
    compact_columns, _ = select_training_features(X.loc[train_ids], drug)
    X = X[compact_columns]
    train_class_ids = y.loc[train_ids].dropna(subset=["binary_class"]).index
    test_class_ids = y.loc[test_ids].dropna(subset=["binary_class"]).index
    specifications = {
        "lineage_only": {"lineage"},
        "expression_only": {"expression"},
        "expression_plus_lineage": {"expression", "lineage"},
        "mutation_plus_copy_number": {"mutations", "copy_number", "signatures"},
        "all_modalities": {"expression", "mutations", "copy_number", "signatures", "lineage"},
        "all_without_lineage": {"expression", "mutations", "copy_number", "signatures"},
    }
    rows: list[dict[str, float | str]] = []
    for name, modalities in specifications.items():
        columns = _columns_for_modalities(X, modalities)
        if not columns:
            continue
        regression, classification = train_xgboost(
            X.loc[train_ids, columns],
            y.loc[train_ids, "auc"],
            y.loc[train_class_ids, "binary_class"].astype(int),
            regression_params=model_params,
            classification_params=model_params,
            seed=seed,
        )
        reg = regression_metrics(
            y.loc[test_ids, "auc"], regression.predict(X.loc[test_ids, columns])
        )
        probs = classification.predict_proba(X.loc[test_class_ids, columns])[:, 1]
        cls = classification_metrics(y.loc[test_class_ids, "binary_class"].astype(int), probs)
        rows.append(
            {
                "ablation": name,
                "n_features": float(len(columns)),
                **{f"regression_{k}": v for k, v in reg.items()},
                **{f"classification_{k}": v for k, v in cls.items()},
            }
        )
    return pd.DataFrame(rows)


def leave_one_lineage_out(
    features: pd.DataFrame,
    response: pd.DataFrame,
    metadata: pd.DataFrame,
    drug: str,
    min_lineage_models: int = 25,
    max_lineages: int = 4,
    seed: int = 42,
    model_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Train without each major lineage and evaluate only on it."""
    labels = labels_for_drug(response, drug)
    common = features.index.intersection(labels.index).intersection(metadata.index)
    X_full = features.loc[common]
    y = labels.loc[common]
    lineage = metadata.loc[common, "lineage"].fillna("unknown").astype(str)
    candidates = lineage.value_counts()
    candidates = candidates[candidates >= min_lineage_models].head(max_lineages)
    rows: list[dict[str, float | str]] = []
    for held_out, _count in candidates.items():
        test_ids = lineage[lineage.eq(held_out)].index
        train_ids = lineage[~lineage.eq(held_out)].index
        train_class_ids = y.loc[train_ids].dropna(subset=["binary_class"]).index
        test_class_ids = y.loc[test_ids].dropna(subset=["binary_class"]).index
        if len(train_class_ids) < 30 or len(test_class_ids) < 8:
            continue
        if (
            y.loc[train_class_ids, "binary_class"].nunique() < 2
            or y.loc[test_class_ids, "binary_class"].nunique() < 2
        ):
            continue
        columns, _ = select_training_features(X_full.loc[train_ids], drug)
        X = X_full[columns]
        regression, classification = train_xgboost(
            X.loc[train_ids],
            y.loc[train_ids, "auc"],
            y.loc[train_class_ids, "binary_class"].astype(int),
            regression_params=model_params,
            classification_params=model_params,
            seed=seed,
        )
        reg_pred = regression.predict(X.loc[test_ids])
        cls_prob = classification.predict_proba(X.loc[test_class_ids])[:, 1]
        reg = regression_metrics(y.loc[test_ids, "auc"], reg_pred)
        cls = classification_metrics(y.loc[test_class_ids, "binary_class"].astype(int), cls_prob)
        transformed_train = regression.named_steps["preprocess"].transform(X.loc[train_ids])
        transformed_test = regression.named_steps["preprocess"].transform(X.loc[test_ids])
        detector = PCADistanceOOD(
            n_components=min(8, transformed_train.shape[1], len(train_ids) - 1)
        ).fit(transformed_train)
        ood_labels = detector.label(transformed_test)
        rows.append(
            {
                "held_out_lineage": held_out,
                "n_train": float(len(train_ids)),
                "n_test": float(len(test_ids)),
                "n_features": float(len(columns)),
                "ood_fraction": float(np.mean(np.asarray(ood_labels) == "out-of-distribution")),
                **{f"regression_{k}": v for k, v in reg.items()},
                **{f"classification_{k}": v for k, v in cls.items()},
            }
        )
    return pd.DataFrame(rows)


def feature_stability(
    features: pd.DataFrame,
    response: pd.DataFrame,
    drug: str,
    n_splits: int = 3,
    top_n: int = 20,
    seed: int = 42,
    model_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Measure how often top SHAP features recur across training folds."""
    labels = labels_for_drug(response, drug).dropna(subset=["binary_class"])
    common = features.index.intersection(labels.index)
    X_full = features.loc[common]
    y = labels.loc[common, "binary_class"].astype(int)
    if len(X_full) < n_splits * 20:
        raise ValueError("Feature stability requires more observations")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    counter: Counter[str] = Counter()
    rank_totals: Counter[str] = Counter()
    for fold, (train_position, test_position) in enumerate(splitter.split(X_full), start=1):
        train_ids = X_full.index[train_position]
        test_ids = X_full.index[test_position]
        columns, _ = select_training_features(X_full.loc[train_ids], drug)
        X = X_full[columns]
        classifier = train_xgboost_classifier(
            X.loc[train_ids],
            y.loc[train_ids],
            params=model_params,
            seed=seed + fold,
        )
        importance = global_importance(classifier, X.loc[test_ids]).head(top_n)
        for rank, feature in enumerate(importance["feature"], start=1):
            counter[str(feature)] += 1
            rank_totals[str(feature)] += rank
    rows = [
        {
            "feature": feature,
            "fold_frequency": frequency,
            "frequency_fraction": frequency / n_splits,
            "mean_rank_when_selected": rank_totals[feature] / frequency,
            "stable": frequency == n_splits,
        }
        for feature, frequency in counter.items()
    ]
    return pd.DataFrame(rows).sort_values(
        ["fold_frequency", "mean_rank_when_selected"], ascending=[False, True]
    )

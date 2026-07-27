"""Evaluation functions shared by training reports and tests."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return robust regression metrics."""
    y_true = np.asarray(true, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if np.allclose(np.nanstd(y_true), 0.0) or np.allclose(np.nanstd(y_pred), 0.0):
        spearman = 0.0
        pearson = 0.0
    else:
        spearman = spearmanr(y_true, y_pred, nan_policy="omit").statistic
        pearson = pearsonr(y_true, y_pred).statistic
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(spearman) if np.isfinite(spearman) else 0.0,
        "pearson": float(pearson) if np.isfinite(pearson) else 0.0,
    }


def classification_metrics(
    true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Return discrimination, threshold and calibration metrics."""
    y_true = np.asarray(true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    predicted = (probs >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "brier": float(brier_score_loss(y_true, probs)),
    }
    metrics["auroc"] = (
        float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) == 2 else float("nan")
    )
    metrics["auprc"] = (
        float(average_precision_score(y_true, probs))
        if len(np.unique(y_true)) == 2
        else float("nan")
    )
    return metrics

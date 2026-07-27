"""Publication-ready performance tables and figures for the GitHub portfolio."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _stem(drug: str) -> str:
    return drug.lower().replace(" ", "_")


def _save(fig: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def bootstrap_auroc(
    true: np.ndarray, probabilities: np.ndarray, n_bootstrap: int = 300, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    true = np.asarray(true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    values: list[float] = []
    for _ in range(n_bootstrap):
        positions = rng.integers(0, len(true), len(true))
        sampled_true = true[positions]
        if np.unique(sampled_true).size < 2:
            continue
        values.append(float(roc_auc_score(sampled_true, probabilities[positions])))
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def generate_internal_figures(
    model_directory: str | Path,
    output_directory: str | Path,
    drugs: list[str],
) -> pd.DataFrame:
    """Generate ROC, accuracy, calibration, regression and comparison figures."""
    model_dir = Path(model_directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    fig_roc, ax_roc = plt.subplots(figsize=(7.2, 5.6))
    fig_pr, ax_pr = plt.subplots(figsize=(7.2, 5.6))
    fig_acc, ax_acc = plt.subplots(figsize=(7.2, 5.6))
    fig_cal, ax_cal = plt.subplots(figsize=(7.2, 5.6))

    thresholds = np.linspace(0.0, 1.0, 101)
    threshold_rows: list[dict[str, float | str]] = []
    for drug in drugs:
        stem = _stem(drug)
        cls_path = model_dir / f"{stem}_internal_classification_predictions.csv"
        reg_path = model_dir / f"{stem}_internal_regression_predictions.csv"
        metrics_path = model_dir / f"{stem}_metrics.json"
        if not cls_path.exists() or not reg_path.exists() or not metrics_path.exists():
            continue
        cls = pd.read_csv(cls_path)
        reg = pd.read_csv(reg_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        true = cls["true_sensitive"].to_numpy(dtype=int)
        probs = cls["sensitivity_probability"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(true, probs)
        auroc = float(roc_auc_score(true, probs))
        lower_ci, upper_ci = bootstrap_auroc(true, probs)
        ax_roc.plot(fpr, tpr, label=f"{drug.title()} (AUC {auroc:.2f})")
        precision, recall, _ = precision_recall_curve(true, probs)
        ax_pr.plot(recall, precision, label=drug.title())
        fraction_positive, mean_predicted = calibration_curve(true, probs, n_bins=6, strategy="quantile")
        ax_cal.plot(mean_predicted, fraction_positive, marker="o", label=drug.title())

        accuracy_values = []
        balanced_values = []
        for threshold in thresholds:
            predicted = (probs >= threshold).astype(int)
            accuracy = accuracy_score(true, predicted)
            balanced = balanced_accuracy_score(true, predicted)
            accuracy_values.append(accuracy)
            balanced_values.append(balanced)
            threshold_rows.append(
                {
                    "drug": drug,
                    "threshold": float(threshold),
                    "accuracy": float(accuracy),
                    "balanced_accuracy": float(balanced),
                }
            )
        ax_acc.plot(thresholds, balanced_values, label=drug.title())

        fig_reg, ax_reg = plt.subplots(figsize=(6.2, 5.6))
        ax_reg.scatter(reg["true_auc"], reg["predicted_auc"], alpha=0.75)
        limits = [
            min(reg["true_auc"].min(), reg["predicted_auc"].min()),
            max(reg["true_auc"].max(), reg["predicted_auc"].max()),
        ]
        ax_reg.plot(limits, limits, linestyle="--")
        ax_reg.set_xlabel("Measured PRISM AUC")
        ax_reg.set_ylabel("Predicted PRISM AUC")
        ax_reg.set_title(f"{drug.title()}: measured vs predicted response")
        _save(fig_reg, output / f"{stem}_regression_scatter.png")

        summary_rows.append(
            {
                "drug": drug,
                "n_models": metrics.get("n_models"),
                "n_features": metrics.get("n_selected_features"),
                "regression_spearman": metrics["regression"]["spearman"],
                "regression_r2": metrics["regression"]["r2"],
                "classification_auroc": metrics["classification"]["auroc"],
                "classification_auroc_ci_lower": lower_ci,
                "classification_auroc_ci_upper": upper_ci,
                "classification_accuracy": metrics["classification"]["accuracy"],
                "classification_balanced_accuracy": metrics["classification"]["balanced_accuracy"],
                "classification_auprc": metrics["classification"]["auprc"],
                "brier": metrics["classification"]["brier"],
                "interval_coverage": metrics["interval_coverage"],
                "elastic_net_auroc": metrics["baselines"]["elastic_net_classification"]["auroc"],
                "lineage_auroc": metrics["baselines"].get("lineage_classification", {}).get("auroc", np.nan),
                "elastic_net_spearman": metrics["baselines"]["elastic_net_regression"]["spearman"],
                "lineage_spearman": metrics["baselines"].get("lineage_regression", {}).get("spearman", np.nan),
            }
        )

    ax_roc.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax_roc.set_xlabel("False-positive rate")
    ax_roc.set_ylabel("True-positive rate")
    ax_roc.set_title("Internal held-out test ROC curves")
    ax_roc.legend(fontsize=8)
    _save(fig_roc, output / "internal_roc_curves.png")

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Internal held-out precision-recall curves")
    ax_pr.legend(fontsize=8)
    _save(fig_pr, output / "internal_precision_recall_curves.png")

    ax_acc.set_xlabel("Sensitivity probability threshold")
    ax_acc.set_ylabel("Balanced accuracy")
    ax_acc.set_title("Accuracy across decision thresholds")
    ax_acc.legend(fontsize=8)
    _save(fig_acc, output / "internal_accuracy_threshold_curves.png")

    ax_cal.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax_cal.set_xlabel("Mean predicted probability")
    ax_cal.set_ylabel("Observed sensitive fraction")
    ax_cal.set_title("Probability calibration on held-out test data")
    ax_cal.legend(fontsize=8)
    _save(fig_cal, output / "internal_calibration_curves.png")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "internal_performance_summary.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output / "accuracy_by_threshold.csv", index=False)

    if not summary.empty:
        positions = np.arange(len(summary))
        width = 0.25
        fig_cmp, ax_cmp = plt.subplots(figsize=(8.0, 5.6))
        ax_cmp.bar(positions - width, summary["lineage_auroc"], width, label="Lineage only")
        ax_cmp.bar(positions, summary["elastic_net_auroc"], width, label="Elastic Net")
        ax_cmp.bar(positions + width, summary["classification_auroc"], width, label="XGBoost")
        ax_cmp.set_xticks(positions)
        ax_cmp.set_xticklabels([str(value).title() for value in summary["drug"]], rotation=25, ha="right")
        ax_cmp.set_ylabel("Held-out AUROC")
        ax_cmp.set_ylim(0, 1.05)
        ax_cmp.set_title("Does molecular XGBoost beat simpler baselines?")
        ax_cmp.legend()
        _save(fig_cmp, output / "internal_model_comparison_auroc.png")
    return summary


def generate_external_figures(
    predictions_by_drug: dict[str, pd.DataFrame],
    output_directory: str | Path,
    suffix: str,
) -> pd.DataFrame:
    """Generate external GDSC2 ROC and response-correlation plots."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    fig_roc, ax_roc = plt.subplots(figsize=(7.2, 5.6))
    fig_acc, ax_acc = plt.subplots(figsize=(7.2, 5.6))
    threshold_rows: list[dict[str, float | str]] = []
    rows: list[dict[str, object]] = []
    for drug, predictions in predictions_by_drug.items():
        tails = predictions[predictions["external_class"].isin(["sensitive", "resistant"])].copy()
        if tails.empty:
            continue
        true = tails["external_class"].eq("sensitive").astype(int).to_numpy()
        probs = tails["sensitivity_probability"].to_numpy(dtype=float)
        if np.unique(true).size < 2:
            continue
        fpr, tpr, _ = roc_curve(true, probs)
        auroc = float(roc_auc_score(true, probs))
        lower_ci, upper_ci = bootstrap_auroc(true, probs)
        ax_roc.plot(fpr, tpr, label=f"{drug.title()} (AUC {auroc:.2f})")
        thresholds = np.linspace(0.0, 1.0, 101)
        balanced_values = []
        for threshold in thresholds:
            predicted = (probs >= threshold).astype(int)
            balanced = balanced_accuracy_score(true, predicted)
            balanced_values.append(balanced)
            threshold_rows.append({"drug": drug, "threshold": float(threshold), "balanced_accuracy": float(balanced)})
        ax_acc.plot(thresholds, balanced_values, label=drug.title())
        rows.append(
            {
                "drug": drug,
                "n_models": len(predictions),
                "n_tail_models": len(tails),
                "auroc": auroc,
                "auroc_ci_lower": lower_ci,
                "auroc_ci_upper": upper_ci,
            }
        )
        fig_scatter, ax_scatter = plt.subplots(figsize=(6.2, 5.6))
        ax_scatter.scatter(predictions["external_response"], predictions["predicted_auc"], alpha=0.75)
        ax_scatter.set_xlabel("Measured GDSC2 AUC")
        ax_scatter.set_ylabel("PRISM-trained predicted AUC")
        ax_scatter.set_title(f"{drug.title()}: external assay transfer ({suffix})")
        _save(fig_scatter, output / f"{_stem(drug)}_{suffix}_external_scatter.png")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax_roc.set_xlabel("False-positive rate")
    ax_roc.set_ylabel("True-positive rate")
    ax_roc.set_title(f"GDSC2 external ROC curves: {suffix.replace('_', ' ')}")
    ax_roc.legend(fontsize=8)
    _save(fig_roc, output / f"external_roc_curves_{suffix}.png")
    ax_acc.set_xlabel("Sensitivity probability threshold")
    ax_acc.set_ylabel("Balanced accuracy")
    ax_acc.set_title(f"GDSC2 accuracy across thresholds: {suffix.replace('_', ' ')}")
    ax_acc.legend(fontsize=8)
    _save(fig_acc, output / f"external_accuracy_threshold_curves_{suffix}.png")
    pd.DataFrame(threshold_rows).to_csv(output / f"external_accuracy_by_threshold_{suffix}.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / f"external_roc_summary_{suffix}.csv", index=False)
    return summary


def generate_oof_figures(
    prediction_directory: str | Path,
    output_directory: str | Path,
    drugs: list[str],
) -> pd.DataFrame:
    """Generate robust five-fold OOF ROC, accuracy and regression figures."""
    source = Path(prediction_directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    fig_roc, ax_roc = plt.subplots(figsize=(7.2, 5.6))
    fig_acc, ax_acc = plt.subplots(figsize=(7.2, 5.6))
    fig_pr, ax_pr = plt.subplots(figsize=(7.2, 5.6))
    thresholds = np.linspace(0.0, 1.0, 101)
    rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, float | str]] = []
    for drug in drugs:
        cls_path = source / f"{_stem(drug)}_oof_classification_predictions.csv"
        reg_path = source / f"{_stem(drug)}_oof_regression_predictions.csv"
        metrics_path = source / f"{_stem(drug)}_oof_metrics.json"
        if not cls_path.exists() or not reg_path.exists() or not metrics_path.exists():
            continue
        cls = pd.read_csv(cls_path)
        reg = pd.read_csv(reg_path)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        true = cls["true_sensitive"].to_numpy(dtype=int)
        probs = cls["sensitivity_probability"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(true, probs)
        auroc = roc_auc_score(true, probs)
        ci_lower, ci_upper = bootstrap_auroc(true, probs)
        ax_roc.plot(fpr, tpr, label=f"{drug.title()} (AUC {auroc:.2f})")
        precision, recall, _ = precision_recall_curve(true, probs)
        ax_pr.plot(recall, precision, label=drug.title())
        balanced_values = []
        for threshold in thresholds:
            predicted = (probs >= threshold).astype(int)
            balanced = balanced_accuracy_score(true, predicted)
            balanced_values.append(balanced)
            threshold_rows.append(
                {"drug": drug, "threshold": float(threshold), "balanced_accuracy": float(balanced)}
            )
        ax_acc.plot(thresholds, balanced_values, label=drug.title())
        rows.append(
            {
                "drug": drug,
                "n_regression": payload["n_regression"],
                "n_classification": payload["n_classification"],
                **{f"regression_{key}": value for key, value in payload["regression"].items()},
                **{f"classification_{key}": value for key, value in payload["classification"].items()},
                "classification_auroc_ci_lower": ci_lower,
                "classification_auroc_ci_upper": ci_upper,
            }
        )
        fig_reg, ax_reg = plt.subplots(figsize=(6.2, 5.6))
        ax_reg.scatter(reg["true_auc"], reg["predicted_auc"], alpha=0.7)
        limits = [min(reg["true_auc"].min(), reg["predicted_auc"].min()), max(reg["true_auc"].max(), reg["predicted_auc"].max())]
        ax_reg.plot(limits, limits, linestyle="--")
        ax_reg.set_xlabel("Measured PRISM AUC")
        ax_reg.set_ylabel("Out-of-fold predicted AUC")
        ax_reg.set_title(f"{drug.title()}: five-fold out-of-fold regression")
        _save(fig_reg, output / f"{_stem(drug)}_oof_regression_scatter.png")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax_roc.set_xlabel("False-positive rate")
    ax_roc.set_ylabel("True-positive rate")
    ax_roc.set_title("Five-fold out-of-fold ROC curves")
    ax_roc.legend(fontsize=8)
    _save(fig_roc, output / "oof_roc_curves.png")
    ax_acc.set_xlabel("Sensitivity probability threshold")
    ax_acc.set_ylabel("Balanced accuracy")
    ax_acc.set_title("Five-fold out-of-fold accuracy across thresholds")
    ax_acc.legend(fontsize=8)
    _save(fig_acc, output / "oof_accuracy_threshold_curves.png")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Five-fold out-of-fold precision-recall curves")
    ax_pr.legend(fontsize=8)
    _save(fig_pr, output / "oof_precision_recall_curves.png")
    summary = pd.DataFrame(rows)
    summary.to_csv(source / "oof_performance_summary.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(source / "oof_accuracy_by_threshold.csv", index=False)
    return summary

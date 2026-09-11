"""Plot saved review evidence without retraining or changing evaluation tables."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports" / "pierre_fabre_review"
OUT = ROOT / "reports" / "figures"
DRUGS = ["trametinib", "afatinib", "palbociclib", "olaparib", "gemcitabine"]
FAMILIES = {
    "xgboost": ("XGBoost", "#293e36"),
    "elastic_net": ("ElasticNet", "#139f9e"),
    "lineage": ("Lineage", "#9bacA6"),
}


def save(fig, name):
    fig.savefig(OUT / name, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    summary = pd.read_csv(DATA / "matched_summary.csv")
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    x = np.arange(len(DRUGS))
    for i, (family, (label, color)) in enumerate(FAMILIES.items()):
        vals = (
            summary.loc[summary.family.eq(family)]
            .set_index("drug")
            .loc[DRUGS, "classification_auroc"]
        )
        bars = ax.bar(x + (i - 1) * 0.25, vals, width=0.24, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    ax.set(
        xticks=x,
        xticklabels=[d.title() for d in DRUGS],
        ylim=(0, 1.04),
        ylabel="AUROC",
        title="Matched five-fold OOF comparison · response extremes",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False)
    save(fig, "review_matched_auroc.png")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    for drug, ax in zip(DRUGS, axes.flat, strict=False):
        frame = pd.read_csv(DATA / f"{drug}_matched_oof.csv").dropna(subset=["true_sensitive"])
        for family, (label, color) in FAMILIES.items():
            group = frame.loc[frame.family.eq(family)]
            fpr, tpr, _ = roc_curve(group.true_sensitive, group.sensitivity_probability)
            auc = summary.loc[
                summary.drug.eq(drug) & summary.family.eq(family), "classification_auroc"
            ].item()
            ax.plot(fpr, tpr, color=color, label=f"{label}: {auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            title=drug.title(),
            xlabel="False positive rate",
            ylabel="True positive rate",
        )
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0,
        0.8,
        "Identical evaluation folds and data roles.\n\nResponse tails are defined using fitting data.\n\nRetrospective development evidence.\n\nCurves do not quantify retraining uncertainty.",
        va="top",
        wrap=True,
    )
    fig.suptitle("Matched out-of-fold ROC curves", fontsize=15)
    save(fig, "review_matched_roc.png")

    budget = pd.read_csv(DATA / "screening_budget_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), layout="constrained")
    for drug, ax in zip(DRUGS[:2], axes, strict=True):
        data = budget.loc[budget.drug.eq(drug)]
        for family, (label, color) in FAMILIES.items():
            group = data.loc[data.family.eq(family)].sort_values("k")
            ax.plot(group.k, group.hits_tie_expected, "o-", label=label, color=color)
            ax.fill_between(
                group.k, group.hits_tie_min, group.hits_tie_max, color=color, alpha=0.15
            )
        random = data.loc[data.family.eq("xgboost")].sort_values("k")
        ax.plot(
            random.k, random.random_expected_hits, "--", color="#b56a14", label="Random expectation"
        )
        ax.set(
            xticks=[5, 10, 20],
            ylim=(0, 20),
            title=f"{drug.title()} · {int(random.n_candidates.iloc[0])} candidates",
            xlabel="Models selected for testing",
            ylabel="Sensitive models recovered",
        )
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Fixed screening budget · all held-out response levels\nUniform boundary-tie expectation; shaded range over tie breaks",
        fontsize=13,
    )
    save(fig, "review_screening_budget.png")

    paired = pd.read_csv(DATA / "paired_auroc_differences.csv")
    paired = paired[paired.baseline.eq("elastic_net")].set_index("drug").loc[DRUGS]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    values = paired.auroc_difference.to_numpy()
    ax.errorbar(
        values,
        np.arange(5),
        xerr=np.vstack([values - paired.ci_lower, paired.ci_upper - values]),
        fmt="o",
        color="#293e36",
        capsize=4,
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set(
        yticks=np.arange(5),
        yticklabels=[d.title() for d in DRUGS],
        xlabel="AUROC difference: XGBoost minus ElasticNet",
        title="Complexity can reduce ranking performance\n95% paired bootstrap intervals conditional on saved OOF scores",
    )
    ax.invert_yaxis()
    fig.supxlabel(
        "Unadjusted exploratory intervals; retraining and split uncertainty excluded", fontsize=10
    )
    save(fig, "review_paired_auroc_difference.png")

    radius = pd.read_csv(DATA / "interval_radius_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for drug, ax in zip(DRUGS[:2], axes, strict=True):
        for family, (label, color) in FAMILIES.items():
            group = radius[radius.drug.eq(drug) & radius.family.eq(family)]
            ax.plot(group.radius_scale, group.selection_fraction, color=color, label=label)
        ax.set(
            title=drug.title(),
            xlabel="Radius / saved 90% radius (per fold)",
            ylabel="Fraction passing interval + probability margin",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Post-hoc policy feasibility · not nominal coverage\nFrozen predictions; status, feature coverage and OOD gates excluded",
        fontsize=12,
    )
    save(fig, "review_interval_feasibility.png")


if __name__ == "__main__":
    main()

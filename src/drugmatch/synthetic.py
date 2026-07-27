"""Deterministic synthetic data with known drug-specific biomarker structure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DRUG_RULES = {
    "trametinib": {"positive": ["EGFR", "PIK3CA"], "negative": ["BRAF", "DUSP6", "SPRY2"]},
    "afatinib": {"positive": ["KRAS", "MET"], "negative": ["EGFR", "ERBB2", "ERBB3"]},
    "palbociclib": {"positive": ["CCNE1", "MYC"], "negative": ["RB1", "CCND1", "CDK4"]},
    "olaparib": {"positive": ["BRCA1", "BRCA2", "RAD51"], "negative": ["PARP1", "POLQ"]},
    "gemcitabine": {"positive": ["CDA"], "negative": ["SLC29A1", "DCK", "RRM1"]},
}


def generate_synthetic_omics(
    output_directory: str | Path,
    n_models: int = 600,
    n_noise_genes: int = 120,
    seed: int = 42,
) -> dict[str, Path]:
    """Generate aligned expression, mutation, CNV, metadata and response files."""
    rng = np.random.default_rng(seed)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    model_ids = [f"ACH-SYN-{index:04d}" for index in range(n_models)]
    lineages = np.array(["lung", "breast", "colorectal", "melanoma", "ovary", "pancreas"])
    lineage = rng.choice(lineages, size=n_models, p=[0.22, 0.2, 0.18, 0.14, 0.14, 0.12])

    biological_genes = sorted(
        {g for rule in DRUG_RULES.values() for group in rule.values() for g in group}
    )
    noise_genes = [f"GENE_{i:03d}" for i in range(n_noise_genes)]
    genes = biological_genes + noise_genes
    expression = pd.DataFrame(
        rng.normal(5.0, 1.2, (n_models, len(genes))), index=model_ids, columns=genes
    )
    mutations = pd.DataFrame(
        rng.binomial(1, 0.08, (n_models, len(biological_genes))),
        index=model_ids,
        columns=biological_genes,
    )
    copy_number = pd.DataFrame(
        rng.normal(0.0, 0.45, (n_models, len(biological_genes))),
        index=model_ids,
        columns=biological_genes,
    )

    # Add plausible lineage patterns and known biomarker enrichment.
    expression.loc[lineage == "melanoma", ["BRAF", "DUSP6", "SPRY2"]] += 1.5
    mutations.loc[lineage == "melanoma", "BRAF"] = rng.binomial(
        1, 0.55, (lineage == "melanoma").sum()
    )
    expression.loc[lineage == "lung", ["EGFR", "ERBB2", "ERBB3"]] += 0.8
    expression.loc[lineage == "breast", ["CCND1", "CDK4"]] += 0.8
    mutations.loc[lineage == "ovary", ["BRCA1", "BRCA2"]] = rng.binomial(
        1, 0.22, ((lineage == "ovary").sum(), 2)
    )

    metadata = pd.DataFrame({"model_id": model_ids, "lineage": lineage}).set_index("model_id")
    responses: list[pd.DataFrame] = []
    lineage_effect = {
        "lung": 0.04,
        "breast": 0.02,
        "colorectal": 0.05,
        "melanoma": -0.03,
        "ovary": 0.0,
        "pancreas": 0.07,
    }
    for drug, rules in DRUG_RULES.items():
        score = np.full(n_models, 0.95)
        for gene in rules["negative"]:
            score -= 0.055 * expression[gene].to_numpy()
            score -= 0.16 * mutations.get(gene, pd.Series(0, index=model_ids)).to_numpy()
            score -= 0.05 * copy_number.get(gene, pd.Series(0, index=model_ids)).to_numpy()
        for gene in rules["positive"]:
            score += 0.045 * expression[gene].to_numpy()
            score += 0.12 * mutations.get(gene, pd.Series(0, index=model_ids)).to_numpy()
        score += np.array([lineage_effect[x] for x in lineage])
        score += rng.normal(0, 0.07, n_models)
        score = (score - score.min()) / (score.max() - score.min())
        responses.append(pd.DataFrame({"model_id": model_ids, "drug": drug, "auc": score}))
    response = pd.concat(responses, ignore_index=True)

    paths = {
        "expression": output / "expression.csv",
        "mutations": output / "mutations.csv",
        "copy_number": output / "copy_number.csv",
        "metadata": output / "metadata.csv",
        "response": output / "response.csv",
    }
    expression.rename_axis("model_id").to_csv(paths["expression"])
    mutations.rename_axis("model_id").to_csv(paths["mutations"])
    copy_number.rename_axis("model_id").to_csv(paths["copy_number"])
    metadata.to_csv(paths["metadata"])
    response.to_csv(paths["response"], index=False)
    return paths

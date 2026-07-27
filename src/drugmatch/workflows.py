"""High-level workflows used by the CLI and reproducibility scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from drugmatch.audit import DrugSelectionCriteria, audit_drugs, select_mvp_drugs
from drugmatch.data import (
    combine_mutation_matrices,
    load_model_metadata,
    load_prism_response,
    load_wide_omics,
)
from drugmatch.preprocessing import OmicsTables, assemble_features
from drugmatch.synthetic import generate_synthetic_omics
from drugmatch.training import TrainingOutcome, train_drug_bundle


PREFERRED_DRUGS = ["trametinib", "afatinib", "palbociclib", "olaparib", "gemcitabine"]


def load_aligned_features(
    expression_path: str | Path,
    metadata_path: str | Path,
    mutation_path: str | Path | None = None,
    copy_number_path: str | Path | None = None,
    hotspot_mutation_path: str | Path | None = None,
    signatures_path: str | Path | None = None,
    model_ids: Sequence[str] | pd.Index | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and align the real molecular modalities at stable DepMap ModelID level."""
    expression = load_wide_omics(expression_path, model_ids=model_ids)
    metadata = load_model_metadata(metadata_path)
    damaging = load_wide_omics(mutation_path, model_ids=model_ids) if mutation_path else None
    hotspot = load_wide_omics(hotspot_mutation_path, model_ids=model_ids) if hotspot_mutation_path else None
    mutations = combine_mutation_matrices(damaging, hotspot)
    copy_number = load_wide_omics(copy_number_path, model_ids=model_ids) if copy_number_path else None
    signatures = load_wide_omics(signatures_path, model_ids=model_ids) if signatures_path else None
    features = assemble_features(
        OmicsTables(expression, mutations, copy_number, metadata, signatures=signatures),
        model_ids=model_ids,
    )
    return features, metadata.loc[features.index]


def run_synthetic_training(root: str | Path, seed: int = 42) -> list[TrainingOutcome]:
    root_path = Path(root)
    fixture_directory = root_path / "data" / "processed" / "synthetic_omics"
    paths = generate_synthetic_omics(fixture_directory, seed=seed)
    expression = load_wide_omics(paths["expression"])
    mutations = load_wide_omics(paths["mutations"])
    copy_number = load_wide_omics(paths["copy_number"])
    metadata = load_model_metadata(paths["metadata"])
    response = pd.read_csv(paths["response"])
    features = assemble_features(OmicsTables(expression, mutations, copy_number, metadata))
    outcomes: list[TrainingOutcome] = []
    for drug in PREFERRED_DRUGS:
        outcomes.append(
            train_drug_bundle(
                drug,
                features,
                response,
                metadata,
                root_path / "models" / "demo" / "synthetic_omics",
                data_release="Deterministic synthetic omics v1",
                seed=seed,
                tune=False,
            )
        )
    return outcomes


def run_prism_lineage_demo(
    root: str | Path,
    response_path: str | Path,
    metadata_path: str | Path,
    seed: int = 42,
) -> list[TrainingOutcome]:
    root_path = Path(root)
    response = load_prism_response(response_path).rename(columns={"depmap_id": "model_id"})
    metadata = load_model_metadata(metadata_path)
    audit = audit_drugs(response, metadata, DrugSelectionCriteria())
    audit.to_csv(root_path / "reports" / "data_audit" / "prism_drug_audit.csv", index=False)
    selected = select_mvp_drugs(audit, preferred=PREFERRED_DRUGS)
    expression = pd.DataFrame({"baseline": 0.0}, index=metadata.index)
    features = assemble_features(OmicsTables(expression, None, None, metadata))
    outcomes: list[TrainingOutcome] = []
    for drug in selected:
        outcomes.append(
            train_drug_bundle(
                drug,
                features,
                response,
                metadata,
                root_path / "models" / "demo" / "prism_lineage",
                data_release="PRISM Repurposing Secondary Screen + PRISM lineage metadata",
                seed=seed,
                tune=False,
            )
        )
    return outcomes

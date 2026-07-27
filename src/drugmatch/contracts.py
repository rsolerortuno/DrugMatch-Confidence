"""Validated contracts separating data preparation from model training."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DataProvenance(BaseModel):
    """Provenance attached to processed datasets and trained models."""

    model_config = ConfigDict(extra="forbid")

    source: str
    release: str
    files: dict[str, str] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class DatasetSplit(BaseModel):
    """Immutable grouped split membership."""

    model_config = ConfigDict(extra="forbid")

    train_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]

    @field_validator("validation_ids", "test_ids")
    @classmethod
    def no_duplicate_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Split contains duplicate model identifiers")
        return value

    def assert_disjoint(self) -> None:
        train = set(self.train_ids)
        valid = set(self.validation_ids)
        test = set(self.test_ids)
        if train & valid or train & test or valid & test:
            raise ValueError("Train, validation and test identifiers must be disjoint")


class FeatureManifest(BaseModel):
    """Features expected by a serialized model."""

    model_config = ConfigDict(extra="forbid")

    drug: str
    expression_features: list[str] = Field(default_factory=list)
    mutation_features: list[str] = Field(default_factory=list)
    copy_number_features: list[str] = Field(default_factory=list)
    lineage_categories: list[str] = Field(default_factory=list)
    transformed_features: list[str] = Field(default_factory=list)


class ModelBundleMetadata(BaseModel):
    """Version and intended-use metadata bundled with a model."""

    model_config = ConfigDict(extra="forbid")

    project_version: str
    model_version: str
    drug: str
    task: Literal["regression", "classification"]
    data_release: str
    preclinical_only: bool = True
    feature_manifest_path: str
    disclaimer: str


class ProcessedDataset:
    """Runtime contract for aligned features, labels and metadata."""

    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        metadata: pd.DataFrame,
    ) -> None:
        if not features.index.is_unique:
            raise ValueError("Feature matrix index must be unique")
        if not labels.index.is_unique:
            raise ValueError("Label matrix index must be unique")
        if not metadata.index.is_unique:
            raise ValueError("Metadata index must be unique")
        common = features.index.intersection(labels.index).intersection(metadata.index)
        if len(common) == 0:
            raise ValueError("Features, labels and metadata do not share model identifiers")
        self.features = features.loc[common].copy()
        self.labels = labels.loc[common].copy()
        self.metadata = metadata.loc[common].copy()

    def save(self, directory: str | Path) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        self.features.to_csv(output / "features.csv")
        self.labels.to_csv(output / "labels.csv")
        self.metadata.to_csv(output / "metadata.csv")

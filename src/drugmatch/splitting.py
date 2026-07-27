"""Leakage-safe grouped train, validation and test splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from drugmatch.contracts import DatasetSplit


def grouped_split(
    model_ids: list[str] | pd.Index,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    stratify: pd.Series | None = None,
) -> DatasetSplit:
    """Split unique cell-line identifiers into immutable disjoint partitions."""
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("Split fractions must sum to one")
    ids = pd.Index(model_ids).drop_duplicates().astype(str)
    if len(ids) < 10:
        raise ValueError("At least ten unique model identifiers are required")
    strat_values = None
    if stratify is not None:
        strat_values = stratify.reindex(ids).map(
            lambda value: "unknown" if pd.isna(value) else str(value)
        )
        if strat_values.value_counts().min() < 2:
            strat_values = None
    train_ids, remaining = train_test_split(
        ids,
        train_size=train_fraction,
        random_state=seed,
        stratify=strat_values,
    )
    remaining_fraction = validation_fraction + test_fraction
    validation_share = validation_fraction / remaining_fraction
    remaining_strat = None
    if strat_values is not None:
        remaining_strat = strat_values.reindex(remaining)
        if remaining_strat.value_counts().min() < 2:
            remaining_strat = None
    valid_ids, test_ids = train_test_split(
        remaining,
        train_size=validation_share,
        random_state=seed + 1,
        stratify=remaining_strat,
    )
    split = DatasetSplit(
        train_ids=list(map(str, train_ids)),
        validation_ids=list(map(str, valid_ids)),
        test_ids=list(map(str, test_ids)),
    )
    split.assert_disjoint()
    return split


def split_frame(frame: pd.DataFrame, split: DatasetSplit) -> dict[str, pd.DataFrame]:
    """Apply stored model membership to a model-indexed frame."""
    return {
        "train": frame.loc[frame.index.intersection(split.train_ids)].copy(),
        "validation": frame.loc[frame.index.intersection(split.validation_ids)].copy(),
        "test": frame.loc[frame.index.intersection(split.test_ids)].copy(),
    }

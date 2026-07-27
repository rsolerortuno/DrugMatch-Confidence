"""Leakage-safe grouped train, validation and test splits."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from drugmatch.contracts import DatasetSplit


def grouped_split(
    model_ids: Sequence[str] | pd.Index,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    stratify: pd.Series | None = None,
) -> DatasetSplit:
    """Split unique cell-line identifiers into immutable disjoint partitions."""
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("Split fractions must sum to one")

    # Convert explicitly to NumPy before passing values to scikit-learn.
    # This avoids indexing failures with pandas Arrow-backed indexes.
    ids_index = pd.Index(model_ids).drop_duplicates().astype(str)
    ids = np.asarray(ids_index.tolist(), dtype=object)

    if len(ids) < 10:
        raise ValueError("At least ten unique model identifiers are required")

    strat_values: np.ndarray | None = None

    if stratify is not None:
        aligned_stratify = stratify.reindex(ids_index).map(
            lambda value: "unknown" if pd.isna(value) else str(value)
        )

        if not aligned_stratify.empty and aligned_stratify.value_counts().min() >= 2:
            strat_values = np.asarray(aligned_stratify.tolist(), dtype=object)

    if strat_values is None:
        train_ids, remaining = train_test_split(
            ids,
            train_size=train_fraction,
            random_state=seed,
        )
        remaining_strat = None
    else:
        train_ids, remaining, _, remaining_strat = train_test_split(
            ids,
            strat_values,
            train_size=train_fraction,
            random_state=seed,
            stratify=strat_values,
        )

    remaining_fraction = validation_fraction + test_fraction
    validation_share = validation_fraction / remaining_fraction

    if remaining_strat is not None:
        _, class_counts = np.unique(remaining_strat, return_counts=True)
        if len(class_counts) == 0 or class_counts.min() < 2:
            remaining_strat = None

    valid_ids, test_ids = train_test_split(
        remaining,
        train_size=validation_share,
        random_state=seed + 1,
        stratify=remaining_strat,
    )

    split = DatasetSplit(
        train_ids=[str(value) for value in train_ids],
        validation_ids=[str(value) for value in valid_ids],
        test_ids=[str(value) for value in test_ids],
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

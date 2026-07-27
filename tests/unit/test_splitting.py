import pandas as pd

from drugmatch.splitting import grouped_split


def test_grouped_split_is_disjoint_and_complete() -> None:
    ids = [f"M{i:03d}" for i in range(100)]
    lineage = pd.Series(["a"] * 50 + ["b"] * 50, index=ids)
    split = grouped_split(ids, stratify=lineage)
    split.assert_disjoint()
    combined = set(split.train_ids + split.validation_ids + split.test_ids)
    assert combined == set(ids)

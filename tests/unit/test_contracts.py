import pytest

from drugmatch.contracts import DatasetSplit


def test_split_detects_overlap() -> None:
    split = DatasetSplit(train_ids=["A"], validation_ids=["B"], test_ids=["A"])
    with pytest.raises(ValueError, match="disjoint"):
        split.assert_disjoint()

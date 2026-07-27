import pandas as pd
import pytest

from drugmatch.utils import ensure_unique_index, normalize_name


def test_normalize_name() -> None:
    assert normalize_name("NCI-H1975 / Lung") == "NCIH1975LUNG"


def test_duplicate_index_fails() -> None:
    frame = pd.DataFrame({"x": [1, 2]}, index=["A", "A"])
    with pytest.raises(ValueError, match="duplicate"):
        ensure_unique_index(frame, "frame")

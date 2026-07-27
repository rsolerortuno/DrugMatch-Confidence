from pathlib import Path

import pandas as pd
import pytest

from drugmatch.data import combine_mutation_matrices, load_prism_response, load_wide_omics


def test_load_normalized_response(tmp_path: Path) -> None:
    path = tmp_path / "response.csv"
    pd.DataFrame(
        {
            "model_id": ["A", "A", "B"],
            "drug": ["Drug X", "Drug X", "Drug X"],
            "auc": [0.2, 0.4, 0.8],
        }
    ).to_csv(path, index=False)
    result = load_prism_response(path)
    assert list(result.columns) == ["model_id", "drug", "auc"]
    assert result.loc[result["model_id"].eq("A"), "auc"].iloc[0] == pytest.approx(0.3)


def test_depmap_yes_flag_and_technical_columns_are_handled(tmp_path: Path) -> None:
    path = tmp_path / "omics.csv"
    pd.DataFrame(
        {
            "ProfileID": ["PR-1", "PR-2", "PR-3"],
            "ModelID": ["ACH-1", "ACH-1", "ACH-2"],
            "IsDefaultEntryForModel": ["No", "Yes", "Yes"],
            "SequencingID": ["S1", "S2", "S3"],
            "GENE1": [99.0, 1.5, 2.5],
        }
    ).to_csv(path, index=False)
    result = load_wide_omics(path)
    assert list(result.index) == ["ACH-1", "ACH-2"]
    assert list(result.columns) == ["GENE1"]
    assert result.loc["ACH-1", "GENE1"] == pytest.approx(1.5)
    assert str(result.dtypes.iloc[0]) == "float32"


def test_damaging_and_hotspot_mutations_remain_distinct() -> None:
    damaging = pd.DataFrame({"BRAF": [0, 1]}, index=["A", "B"])
    hotspot = pd.DataFrame({"BRAF": [1, 0]}, index=["A", "B"])
    result = combine_mutation_matrices(damaging, hotspot)
    assert result is not None
    assert set(result.columns) == {"damaging::BRAF", "hotspot::BRAF"}
    assert result.loc["A", "hotspot::BRAF"] == 1
    assert result.loc["A", "damaging::BRAF"] == 0

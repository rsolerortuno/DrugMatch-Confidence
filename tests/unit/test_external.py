import pandas as pd
import pytest

from drugmatch.external import map_gdsc_models, standardize_external_response


def test_gdsc_mapping_excludes_ambiguous_sanger_ids() -> None:
    metadata = pd.DataFrame(
        {
            "model_id": ["ACH-1", "ACH-2", "ACH-3"],
            "SangerModelID": ["SIDM1", "SIDM2", "SIDM2"],
            "CellLineName": ["A", "B", "C"],
            "COSMICID": [1, 2, 3],
        }
    ).set_index("model_id")
    gdsc = pd.DataFrame(
        {
            "SANGER_MODEL_ID": ["SIDM1", "SIDM2"],
            "DRUG_NAME": ["trametinib", "trametinib"],
            "AUC": [0.2, 0.8],
        }
    )
    mapped, audit = map_gdsc_models(gdsc, metadata)
    assert mapped.loc[0, "model_id"] == "ACH-1"
    assert pd.isna(mapped.loc[1, "model_id"])
    assert audit.loc[0, "n_ambiguous_sanger_ids"] == 1


def test_external_response_is_standardized_by_drug() -> None:
    frame = pd.DataFrame(
        {
            "model_id": ["A", "A", "B"],
            "DRUG_NAME": ["Afatinib", "Afatinib", "Other"],
            "AUC": [0.2, 0.4, 0.8],
        }
    )
    result = standardize_external_response(frame, "afatinib")
    assert list(result.columns) == ["model_id", "external_response"]
    assert result.loc[0, "external_response"] == pytest.approx(0.3)

from pathlib import Path

import pandas as pd

from drugmatch.workflows import load_aligned_features


def test_aligned_features_include_signatures_and_call_types(tmp_path: Path) -> None:
    ids = ["ACH-1", "ACH-2", "ACH-3"]
    pd.DataFrame(
        {
            "ModelID": ids,
            "IsDefaultEntryForModel": ["Yes"] * 3,
            "GENE_A": [1.0, 2.0, 3.0],
        }
    ).to_csv(tmp_path / "expression.csv", index=False)
    pd.DataFrame({"ModelID": ids, "TP53": [1, 0, 1]}).to_csv(tmp_path / "damaging.csv", index=False)
    pd.DataFrame({"ModelID": ids, "BRAF": [0, 1, 0]}).to_csv(tmp_path / "hotspot.csv", index=False)
    pd.DataFrame({"ModelID": ids, "ERBB2": [2.0, 4.0, 1.0]}).to_csv(
        tmp_path / "copy.csv", index=False
    )
    pd.DataFrame({"ModelID": ids, "MSIScore": [0.1, 10.0, 0.2]}).to_csv(
        tmp_path / "signatures.csv", index=False
    )
    pd.DataFrame({"ModelID": ids, "OncotreeLineage": ["Lung", "Skin", "Breast"]}).to_csv(
        tmp_path / "metadata.csv", index=False
    )

    features, metadata = load_aligned_features(
        tmp_path / "expression.csv",
        tmp_path / "metadata.csv",
        mutation_path=tmp_path / "damaging.csv",
        hotspot_mutation_path=tmp_path / "hotspot.csv",
        copy_number_path=tmp_path / "copy.csv",
        signatures_path=tmp_path / "signatures.csv",
    )
    assert set(features.index) == set(ids)
    assert "expr::GENE_A" in features
    assert "mut::damaging::TP53" in features
    assert "mut::hotspot::BRAF" in features
    assert "cn::ERBB2" in features
    assert "sig::MSIScore" in features
    assert "meta::lineage" in features
    assert metadata.loc["ACH-1", "lineage"] == "lung"

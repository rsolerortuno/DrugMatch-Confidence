import pandas as pd

from drugmatch.audit import DrugSelectionCriteria, audit_drugs


def test_audit_accepts_well_covered_drug() -> None:
    ids = [f"M{i}" for i in range(80)]
    response = pd.DataFrame({"model_id": ids, "drug": "x", "auc": [i / 80 for i in range(80)]})
    metadata = pd.DataFrame(
        {"model_id": ids, "lineage": [f"l{i % 5}" for i in range(80)]}
    ).set_index("model_id")
    criteria = DrugSelectionCriteria(
        min_models=50,
        min_sensitive=10,
        min_resistant=10,
        min_lineages=5,
    )
    report = audit_drugs(response, metadata, criteria)
    assert bool(report.iloc[0]["accepted"])

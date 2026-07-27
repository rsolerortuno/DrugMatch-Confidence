import pandas as pd

from drugmatch.labels import labels_for_drug


def test_labels_use_response_tails() -> None:
    frame = pd.DataFrame({"model_id": [f"M{i}" for i in range(40)], "drug": "x", "auc": range(40)})
    labels = labels_for_drug(frame, "x")
    assert (labels["class"] == "sensitive").sum() == 10
    assert (labels["class"] == "resistant").sum() == 10
    assert labels["binary_class"].notna().sum() == 20

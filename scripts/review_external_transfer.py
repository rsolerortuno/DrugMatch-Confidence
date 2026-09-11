"""Retrospective GDSC2 transfer check of the corrected bundles; never tune here."""

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from drugmatch.api import DrugMatchPredictor
from drugmatch.external import standardize_external_response, validate_external

ROOT = Path(__file__).resolve().parents[1]


def main():
    features = joblib.load(ROOT / "data/raw/recovered/aligned_review_cache.joblib")["features"]
    source = ROOT / "reports/external_validation/gdsc2_mapped_selected_columns.csv"
    gdsc = pd.read_csv(source)
    out = ROOT / "reports/pierre_fabre_review"
    rows = []
    for path in sorted((ROOT / "models/review/depmap_26q1_prism").glob("*.joblib")):
        p = DrugMatchPredictor.load(path)
        roles = p.bundle["partition_roles"]
        used = set(roles["fit"]) | set(roles["tuning"]) | set(roles["calibration"])
        assert not used & set(roles["test"])
        # Earlier results have already been inspected. This is re-evaluation,
        # not newly untouched external evidence and cannot select new policies.
        external = standardize_external_response(gdsc, p.bundle["drug"])
        metrics, pred = validate_external(p, features, external, allowed_model_ids=roles["test"])
        assert not set(pred.model_id) & used
        pred.to_csv(out / f"{path.stem}_external_reanalysis.csv", index=False)
        rows.append(dict(drug=path.stem, **metrics))
    pd.DataFrame(rows).to_csv(out / "external_reanalysis_summary.csv", index=False)
    (out / "external_reanalysis_protocol.json").write_text(
        json.dumps(
            dict(
                status="Retrospective re-evaluation of previously inspected GDSC2; no prospective claim.",
                endpoint="AUROC uses external within-drug quartiles; it tests ranking, not transport of PRISM calibration.",
                source_sha256=hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

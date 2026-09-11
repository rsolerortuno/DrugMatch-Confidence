"""Reproduce the September development review from locally supplied source data."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import pandas as pd

from drugmatch.data import load_prism_response
from drugmatch.matched_validation import matched_oof, summarize_matched
from drugmatch.training import train_drug_bundle
from drugmatch.workflows import PREFERRED_DRUGS, load_aligned_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/pierre_fabre_review"))
    parser.add_argument("--drugs", nargs="+", default=PREFERRED_DRUGS)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.raw
    sources = dict(
        expression_path=raw / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv.zip",
        metadata_path=raw / "Model.csv",
        mutation_path=raw / "OmicsSomaticMutationsMatrixDamaging.csv.zip",
        copy_number_path=raw / "PortalOmicsCNGeneLog2.csv.zip",
        hotspot_mutation_path=raw / "OmicsSomaticMutationsMatrixHotspot.csv",
        signatures_path=raw / "OmicsGlobalSignatures.csv",
    )
    response_path = raw / "secondary-screen-dose-response-curve-parameters.csv"
    manifest = {
        p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
        for p in [*sources.values(), response_path]
    }
    response = load_prism_response(response_path).rename(columns={"depmap_id": "model_id"})
    response = response[response.drug.isin(PREFERRED_DRUGS)]
    cache = raw / "aligned_review_cache.joblib"
    if cache.exists():
        cached = joblib.load(cache)
        if cached["source_hashes"] != manifest:
            raise ValueError("Stale source cache")
        features, metadata = cached["features"], cached["metadata"]
    else:
        features, metadata = load_aligned_features(**sources, model_ids=response.model_id.unique())
        joblib.dump(
            dict(source_hashes=manifest, features=features, metadata=metadata), cache, compress=3
        )
    (args.output / "input_manifest.json").write_text(
        json.dumps(
            dict(
                source_sha256=manifest,
                n_feature_models=len(features),
                n_features=len(features.columns),
                response_min=float(response.auc.min()),
                response_max=float(response.auc.max()),
                release="DepMap Public 26Q1 + PRISM secondary screen",
                packages={
                    p: importlib.metadata.version(p)
                    for p in ["numpy", "pandas", "scikit-learn", "xgboost", "shap"]
                },
                protocol="Fixed matched OOF; retrospective development reanalysis; no new external confirmation",
                seed=2026,
                split_stratification="Full-cohort outcome-rank quartiles allocate outer/inner folds only; feature selection, class cutoffs and calibration remain fit/fold-local.",
            ),
            indent=2,
        )
    )
    print("Loaded", features.shape, flush=True)
    all_preds = []
    for drug in args.drugs:
        print("Matched OOF:", drug, flush=True)
        calibration_rows = []
        preds, roles = matched_oof(
            features, response, metadata, drug, calibration_output=calibration_rows
        )
        pd.DataFrame(calibration_rows).to_csv(
            args.output / f"{drug}_calibration_residuals.csv", index=False
        )
        preds.to_csv(args.output / f"{drug}_matched_oof.csv", index=False)
        roles.to_csv(args.output / f"{drug}_fold_roles.csv", index=False)
        summary = summarize_matched(preds)
        summary.to_csv(args.output / f"{drug}_summary.csv", index=False)
        print(
            summary[
                [
                    "drug",
                    "family",
                    "classification_auroc",
                    "regression_spearman",
                    "technical_coverage",
                ]
            ].to_string(index=False),
            flush=True,
        )
        all_preds.append(preds)
        if args.train:
            print("Independent-calibration bundle:", drug, flush=True)
            train_drug_bundle(
                drug,
                features,
                response,
                metadata,
                "models/review/depmap_26q1_prism",
                "DepMap Public 26Q1 + PRISM secondary screen; September retrospective review",
            )
    summarize_matched(pd.concat(all_preds)).to_csv(args.output / "matched_summary.csv", index=False)


if __name__ == "__main__":
    main()

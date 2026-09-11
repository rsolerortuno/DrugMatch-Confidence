# Changelog

## 1.1.0.dev0 — evidence review (2026-09-11)

- Fix the release checker to verify the ten shipped historical/review bundles and an existing example; run it in CI.
- Update README quick start, repository map, citation metadata and release notes.
- Declare PyArrow in development dependencies and run CI quality checks with only `[dev]`, without relying on Streamlit transitively.

- Add per-fold interval feasibility and post-hoc radius sensitivity; distinguish radius from nominal coverage and assay noise.
- Report boundary-tie minimum/maximum/expected screening hits and highlight the conditional gemcitabine complexity penalty.
- Disclose outcome-stratified fold allocation; preserve calibration residuals for future coverage-feasibility audits.


- Separate fit, tune, calibration and test roles; fixed calibration policy and exact conformal rank.
- Explicit sensitive/resistant/abstain decisions with provenance and input-support checks.
- Matched XGBoost, ElasticNet and lineage baselines, saved OOF predictions and screening-budget analysis.
- Five review bundles retained as unreviewed; historical bundles and figures preserved.
- Reproducible review figures in the existing `reports/figures/` directory.
- Fix Arrow-backed identifier indexing in training and both OOF paths; add regression coverage.
- Align Python 3.12 and core dependency pins with the serialized models' training environment.
- CI runs on branch pushes and pull requests, including both historical and review bundle smoke tests.

## 1.0.0 — 2026-07-27

### Added

- Real DepMap Public 26Q1 multimodal training for five drugs.
- Separate damaging and hotspot mutation representations.
- Global genomic signatures and model-level copy number.
- Training-only molecular feature selection.
- Five-fold out-of-fold validation with ROC confidence intervals.
- Frozen GDSC2 external validation and direct assay-concordance analysis.
- ROC, precision-recall, calibration and balanced-accuracy threshold plots.
- Model release labels: validated demo, exploratory and insufficient evidence.
- Real trained bundles, example inputs, non-expert documentation and data manifest.

### Changed

- Version bumped from 0.1.0 to 1.0.0.
- Processed feature export uses pickle by default to avoid an undeclared Parquet-engine dependency.
- Streamlit and API now expose evidence status and summary.

### Scientific outcome

- Trametinib and afatinib are validated portfolio demonstrations.
- Palbociclib is exploratory because strict external transfer is weak.
- Olaparib and gemcitabine are retained as transparent weak/negative results.

## 0.1.0 — 2026-07-27

Initial software-complete release with synthetic full-omics models and real PRISM lineage baselines, pending real molecular files.

# Changelog

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

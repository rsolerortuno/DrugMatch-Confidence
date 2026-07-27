# Test and release verification report

## Release

- Package version: **1.0.0**
- Python package import: passed
- Compiled wheel: `drugmatch_confidence-1.0.0-py3-none-any.whl`
- Automated tests: **22 passed**
- Python byte-code compilation: passed

## Functional smoke tests

- CLI help: passed.
- CLI prediction with the bundled trametinib example: passed.
- All five real `.joblib` bundles load and return a valid probability, clipped AUC interval, response zone, model-agreement status and validation label.
- Synthetic local data build writes `features.pkl` and can be reloaded without a Parquet dependency.
- README local links resolve.
- Embedded ROC and balanced-accuracy images are valid PNG files.

## Data and scientific checks

- Stable DepMap `ModelID` joins are used.
- Default omics profiles are selected from DepMap Yes/No flags.
- Technical identifier columns are excluded from molecular features.
- Damaging and hotspot mutation calls remain distinct.
- Feature selection is fitted using training data only.
- Five-fold OOF validation repeats feature selection and calibration inside each fold.
- GDSC2 is evaluated only after model freezing.

## Release labels

- **Trametinib** — `validated_demo`: Recommended as the main portfolio demonstration.
- **Afatinib** — `validated_demo`: Recommended as the main portfolio demonstration.
- **Palbociclib** — `exploratory`: Useful research signal, but transfer is not sufficiently stable for the main claim.
- **Olaparib** — `insufficient_evidence`: Included as a documented negative/weak result; do not use as a reliable predictor.
- **Gemcitabine** — `insufficient_evidence`: Included as a documented negative/weak result; do not use as a reliable predictor.

## Environment limitations

`ruff` and `mypy` are declared in the development dependencies and executed by GitHub Actions. The local offline package mirror did not provide those executables during this build, so this report does not falsely claim a local lint/type-check pass. A Docker job is also configured in CI; the current runtime had no Docker daemon.

# Architecture

## Components

- `data.py`: public-file ingestion, release-tolerant schemas, manifests and checksums.
- `workflows.py`: aligned multimodal matrix construction.
- `preprocessing.py`: modality prefixes and training-only feature selection.
- `splitting.py`: immutable grouped partitions.
- `models.py`: baselines, Elastic Net and XGBoost pipelines.
- `calibration.py`: Platt and isotonic calibration.
- `uncertainty.py`: split-conformal intervals.
- `ood.py`: PCA-distance OOD scoring.
- `training.py`: drug-specific training, tuning, evaluation and serialization.
- `cross_validation.py`: leakage-safe five-fold OOF validation.
- `external.py`: frozen GDSC2 mapping and evaluation.
- `robustness.py`: lineage holdout, modality ablation and feature stability.
- `reporting.py`: ROC, precision-recall, accuracy-threshold, calibration and scatter figures.
- `api.py`: stable single-sample prediction contract.
- `cli.py`: reproducible command-line workflows.
- `app/streamlit_app.py`: interactive demonstration.

## Model bundle

Each `.joblib` bundle contains the fitted regression and classification pipelines, calibrator, conformal interval, OOD detector, selected feature contract, reference training samples, split metadata, metrics, validation status, evidence summary and preclinical disclaimer.

## Data-leakage boundary

Feature selection, imputation, scaling, class thresholds, calibration and probability threshold selection are fitted only with their appropriate training or validation data. GDSC2 is downstream of the frozen bundle.

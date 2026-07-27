# Model card: DrugMatch-Confidence 1.0.0

## Intended use

Research and portfolio demonstration of interpretable, uncertainty-aware pharmacogenomic machine learning in preclinical cancer cell lines.

## Excluded use

Patient treatment selection, clinical decision support, dosing, diagnosis, prognosis or any medical action.

## Training data

DepMap Public 26Q1 molecular profiles joined to PRISM secondary-screen dose-response AUC by stable `ModelID`. The aligned molecular universe contains 1,105 models and 58,498 raw variables before drug-specific training-only selection.

## Supported drugs and evidence

| Drug | Release status | 5-fold OOF AUROC | OOF balanced accuracy | Strict GDSC2 AUROC | Strict GDSC2 n | Interpretation |
|---|---|---:|---:|---:|---:|---|
| Trametinib | `validated_demo` | 0.86 | 0.78 | 0.84 | 20 | Recommended as the main portfolio demonstration. |
| Afatinib | `validated_demo` | 0.92 | 0.85 | 0.83 | 22 | Recommended as the main portfolio demonstration. |
| Palbociclib | `exploratory` | 0.75 | 0.64 | 0.51 | 22 | Useful research signal, but transfer is not sufficiently stable for the main claim. |
| Olaparib | `insufficient_evidence` | 0.54 | 0.52 | 0.68 | 24 | Included as a documented negative/weak result; do not use as a reliable predictor. |
| Gemcitabine | `insufficient_evidence` | 0.48 | 0.49 | 0.70 | 18 | Included as a documented negative/weak result; do not use as a reliable predictor. |

## Validation

- fixed grouped internal test set;
- leakage-safe five-fold out-of-fold evaluation;
- GDSC2 frozen external validation;
- leave-one-lineage-out stress testing;
- modality ablation and feature-stability analysis;
- bootstrap AUROC confidence intervals.

## Outputs

Continuous response, conformal interval, sensitive/resistant class, continuous-response zone, model agreement, calibrated probability, decision confidence, OOD status, feature coverage, validation status and SHAP drivers.

## Important limitations

Cell-line biology, experimental batch and assay differences, small strict external subsets, fixed supported compounds, imperfect feature coverage and non-causal explanations. Model status must be checked before interpreting a result.

## Data provenance

See `data/manifests/real_training_inputs.json` and the official source URLs in the README.

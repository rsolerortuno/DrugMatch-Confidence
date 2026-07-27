# Technical data guide

For a plain-language explanation, start with [`DATA_FOR_NON_EXPERTS.md`](DATA_FOR_NON_EXPERTS.md).

## Input releases used for version 1.0.0

- DepMap Public 26Q1 molecular data and `Model.csv`.
- PRISM Repurposing Secondary Screen dose-response data.
- GDSC2 fitted dose-response data dated 27 October 2023.
- Cell Model Passports and GDSC compound exports downloaded in July 2026.

Exact local filenames, byte sizes, SHA-256 hashes, official source pages and roles are recorded in `data/manifests/real_training_inputs.json`.

## Model-level molecular inputs

```text
OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv
OmicsSomaticMutationsMatrixDamaging.csv
OmicsSomaticMutationsMatrixHotspot.csv
PortalOmicsCNGeneLog2.csv
OmicsGlobalSignatures.csv
Model.csv
```

The loaders select `IsDefaultEntryForModel` rows when present, drop technical identifiers, cast numerical matrices to `float32`, preserve damaging and hotspot calls as distinct features, and join tables by stable `ModelID`.

## Primary response

PRISM replicate rows are collapsed to median AUC per model and drug after quality filtering. Lower measured AUC means greater sensitivity.

## External response

GDSC2 rows are mapped through `SangerModelID` from DepMap `Model.csv`. Duplicate Sanger identifiers are classified as ambiguous and excluded. GDSC2 is never used for feature selection, tuning, calibration or decision-threshold selection.

## Processed matrix

The complete aligned universe has 1,105 models and 58,498 raw variables. Drug-specific training uses only models with a valid PRISM outcome. Feature prefixes are:

```text
expr::GENE
mut::damaging::GENE
mut::hotspot::GENE
cn::GENE
sig::SIGNATURE
meta::lineage
```

The local processed matrix is stored as `features.pkl`; it is intentionally excluded from Git because it is approximately 248 MB and can be reproduced from the manifest.

## Leakage controls

- grouped model splits;
- class quartiles computed from training outcomes only;
- feature selection fitted within training data and repeated within each OOF fold;
- validation-only calibration and threshold selection;
- frozen internal test and GDSC2 evaluation.

## Data distribution

Large public source files are not redistributed. Code is MIT licensed; data remain subject to their providers' terms and citation requirements.

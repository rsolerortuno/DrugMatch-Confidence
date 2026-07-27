# Reproducing the real-data release

## Download sources

- DepMap Public 26Q1: <https://depmap.org/portal/data_page/?tab=currentRelease>
- PRISM secondary screen: <https://depmap.org/repurposing/>
- GDSC2 and Cell Model Passports: <https://cellmodelpassports.sanger.ac.uk/downloads>

The exact names, hashes and roles used for version 1.0.0 are recorded in `data/manifests/real_training_inputs.json`.

## Build the aligned matrix

```bash
drugmatch data build \
  --expression data/raw/depmap/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv.zip \
  --mutations data/raw/depmap/OmicsSomaticMutationsMatrixDamaging.csv.zip \
  --hotspot-mutations data/raw/depmap/OmicsSomaticMutationsMatrixHotspot.csv \
  --copy-number data/raw/depmap/PortalOmicsCNGeneLog2.csv.zip \
  --signatures data/raw/depmap/OmicsGlobalSignatures.csv \
  --metadata data/raw/depmap/Model.csv \
  --output data/processed/real_26q1
```

This writes `features.pkl`, `metadata.csv` and `build_summary.json`. Pickle is used for the local processed matrix so the core installation does not require an optional Parquet engine.

## Train all five models

```bash
drugmatch train real-release \
  --response data/raw/prism/secondary-screen-dose-response-curve-parameters.csv \
  --expression data/raw/depmap/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv.zip \
  --mutations data/raw/depmap/OmicsSomaticMutationsMatrixDamaging.csv.zip \
  --hotspot-mutations data/raw/depmap/OmicsSomaticMutationsMatrixHotspot.csv \
  --copy-number data/raw/depmap/PortalOmicsCNGeneLog2.csv.zip \
  --signatures data/raw/depmap/OmicsGlobalSignatures.csv \
  --metadata data/raw/depmap/Model.csv
```

## External validation

GDSC2 must be mapped through `SangerModelID` in `Model.csv`. Ambiguous IDs are excluded. Do not adjust the trained bundle after inspecting GDSC2 outcomes.

## Resource notes

The full raw matrix is wide. The loaders use `float32`, load each modality once and reduce each drug to a compact training-selected feature set. The real release was trained without a large GPU; XGBoost uses CPU histogram trees.

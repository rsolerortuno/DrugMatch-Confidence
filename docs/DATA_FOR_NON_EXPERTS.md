# Data files explained for non-experts

DrugMatch-Confidence joins three kinds of information: what a cancer model looks like molecularly, how it responded to a drug in an experiment, and how the same drug behaved in a separate laboratory dataset.

## Why use stable model IDs?

A cell line can have several names. DepMap assigns identifiers such as `ACH-000012`, while Sanger uses identifiers such as `SIDM01067`. The project maps these IDs explicitly. Ambiguous mappings are excluded rather than guessed.

## DepMap Public 26Q1

Official download page: <https://depmap.org/portal/data_page/?tab=currentRelease>

### `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`

This is an RNA-expression table. Each row is a cancer model and each gene column indicates how actively that gene is being transcribed. Values are already processed as log-transformed TPM, so the project does not start from raw sequencing reads.

### `OmicsSomaticMutationsMatrixDamaging.csv`

This matrix records likely damaging or loss-of-function mutations. It is useful for tumour suppressors and DNA-repair genes such as `TP53`, `PTEN`, `BRCA1` and `BRCA2`.

### `OmicsSomaticMutationsMatrixHotspot.csv`

This matrix records recurrent cancer hotspots, which often represent activating oncogene mutations. It is kept separate from damaging mutations so that, for example, an activating `BRAF` hotspot is not treated as a loss-of-function event.

### `PortalOmicsCNGeneLog2.csv`

This table describes gene copy number. Extra copies can amplify an oncogene; losses can remove a tumour suppressor. The release also contained a WGS linear-scale matrix. Both had the same model coverage and near-identical gene rankings after transformation, so the portal log2 matrix was chosen as the primary input.

### `OmicsGlobalSignatures.csv`

This small file contains summary features including microsatellite instability, ploidy, chromosomal instability, whole-genome doubling, loss-of-heterozygosity fraction and aneuploidy.

### `Model.csv`

This is the dictionary of models. It provides stable IDs, lineage, disease labels, Sanger IDs, COSMIC IDs and names used to connect all datasets.

## PRISM secondary screen

Official page: <https://depmap.org/repurposing/>

### Dose-response curve parameters

This is the training outcome. PRISM exposed cell lines to several concentrations of a drug and fitted a dose-response curve. This project uses the fitted response AUC: lower values mean greater sensitivity.

### Cell-line information and treatment information

These files provide identifiers and compound metadata needed for auditing and mapping.

## GDSC2

Official download page: <https://cellmodelpassports.sanger.ac.uk/downloads>

### `GDSC2_fitted_dose_response_27Oct23.xlsx`

This independent dataset contains fitted GDSC2 responses, including `AUC` and `LN_IC50`. It is used only after PRISM training is complete. It never selects features or tunes the model.

### Cell Model Passports and compound annotation exports

These files support reliable Sanger-model and compound mapping across DepMap, PRISM and GDSC2.

## What is included in GitHub?

The repository contains code, compact trained model bundles, predictions, metrics, figures and a manifest with input hashes. It does not redistribute the large source datasets. This keeps the repository practical and respects the original distribution channels.

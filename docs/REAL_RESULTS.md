> Historical v1.0 results. The revised matched evaluation is documented in [PIERRE_FABRE_REVIEW.md](PIERRE_FABRE_REVIEW.md).

# Real-data results

## Training cohort

After aligning DepMap Public 26Q1 molecular profiles with PRISM, the common molecular matrix contained **1,105 models and 58,498 raw variables**. Drug-specific usable sample counts ranged from 298 to 375.

## Five-fold out-of-fold classification

| Drug | Release status | 5-fold OOF AUROC | OOF balanced accuracy | Strict GDSC2 AUROC | Strict GDSC2 n | Interpretation |
|---|---|---:|---:|---:|---:|---|
| Trametinib | `validated_demo` | 0.86 | 0.78 | 0.84 | 20 | Recommended as the main portfolio demonstration. |
| Afatinib | `validated_demo` | 0.92 | 0.85 | 0.83 | 22 | Recommended as the main portfolio demonstration. |
| Palbociclib | `exploratory` | 0.75 | 0.64 | 0.51 | 22 | Useful research signal, but transfer is not sufficiently stable for the main claim. |
| Olaparib | `insufficient_evidence` | 0.54 | 0.52 | 0.68 | 24 | Included as a documented negative/weak result; do not use as a reliable predictor. |
| Gemcitabine | `insufficient_evidence` | 0.48 | 0.49 | 0.70 | 18 | Included as a documented negative/weak result; do not use as a reliable predictor. |

The OOF estimate is preferred over a single test split because it uses every eligible sample exactly once as an unseen prediction while preserving fold-specific feature selection and calibration.

## Independent GDSC2 transfer

All-mapped GDSC2 tail AUROCs were:

| Drug | Mapped models | Tail AUROC | Spearman response correlation |
|---|---:|---:|---:|
| Trametinib | 507 | 0.91 | 0.61 |
| Afatinib | 506 | 0.68 | 0.33 |
| Palbociclib | 508 | 0.70 | 0.31 |
| Olaparib | 507 | 0.66 | 0.26 |
| Gemcitabine | 506 | 0.65 | 0.16 |

The strict PRISM-test-only analysis is smaller but more conservative. Trametinib and afatinib remained strong; palbociclib did not transfer reliably.

## Assay concordance matters

Direct observed PRISM-versus-GDSC response correlations were approximately 0.65 for trametinib, 0.40 for afatinib, 0.29 for palbociclib, 0.23 for olaparib and 0.26 for gemcitabine. This means the two laboratories do not produce interchangeable outcomes, placing a practical ceiling on transfer performance.

## Biological interpretation

### Trametinib

Expression features dominate. Important signals include `DUSP6` and `EREG` in MAPK feedback, `NF1`, and mesenchymal/extracellular-state features such as `FSTL1` and `PLAT`. The model transfers well to GDSC2.

### Afatinib

The strongest signals describe an epithelial and ERBB-linked state: `IRF6`, `CLDN1`, `CDH1`, `FGFBP1`, and `GRB7` copy number. It is the strongest overall internal classifier and transfers well in the strict external subset.

### Palbociclib

`RB1` expression and copy number are stable and biologically plausible. However, strict GDSC2 AUROC is near random, so the bundle is labelled exploratory rather than validated.

### Olaparib

Mutation/copy-number and genomic-instability variables are more useful than expression alone, but overall discrimination remains weak. This likely reflects incomplete representation of homologous-recombination deficiency and assay-specific response variation.

### Gemcitabine

Features such as `RRM2`, damaging `TP53`, `NUPR1` and stress-response genes appear, but cross-fold stability is poor and OOF performance is near random. The model is preserved as a negative result.

## Confidence intervals

ROC confidence intervals are generated with bootstrap resampling. Strict GDSC2 subsets are small, so their intervals are wider than the all-mapped validation. Point estimates should never be read without sample counts and uncertainty.

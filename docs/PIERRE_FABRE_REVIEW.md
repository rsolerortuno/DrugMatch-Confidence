# September 2026 evidence review

This evaluation establishes an auditable way to reject unsupported deployment choices before spending on a larger preclinical programme: it exposes an inoperative categorical policy on the lead-drug panel, unnecessary complexity for the leading examples and an observed complexity penalty for gemcitabine. It also supplies a reproducible ranking workflow for a prospective pilot. These are project-level decisions supported by the current benchmark, not patient-response claims.

## What changed

- Final training now has four disjoint roles: fit, hyperparameter tuning, calibration, and test. The historical test and calibration memberships are preserved; a tuning partition is taken from the original training partition. Class cutoffs and feature selection use fit models only.
- Platt calibration and a 0.50 decision threshold are fixed in advance for this revision. The calibration set no longer chooses between calibrators or hyperparameters. Conformal calibration uses a model already fixed by the fit/tuning data.
- The conformal radius uses the exact finite-sample order statistic, including an infinite interval when the calibration sample cannot support the requested coverage. Calibration guarantees require exchangeability and are marginal, not per-sample or cross-assay guarantees.
- Modality and lineage analyses now derive class cutoffs from their training partitions. Their historical CSVs are archived results, not results of the corrected implementation.
- A new matched five-fold comparison gives XGBoost, Elastic Net and lineage models identical fit/calibration/test memberships and training-defined endpoints. Fixed-model OOF estimates and tuned final bundles remain different estimands.
- The prediction API returns `decision`, `abstain`, and explicit reasons. Model status, independently calibrated bundle provenance, probability distance from the decision threshold, missing features, distribution shift and the full regression interval all gate a class assignment. Confidence labels are policy labels, not probabilities of correctness.
- Assay AUC is no longer forcibly clipped to [0, 1]. The recovered PRISM response values include values above 1, and clipping after conformal calibration made inference inconsistent with the residual calculation.

## Matched OOF results

| Drug | n models / extremes | Lineage AUROC | Linear AUROC | XGBoost AUROC |
|---|---:|---:|---:|---:|
| Trametinib | 375 / 190 | 0.755 | 0.863 | 0.848 |
| Afatinib | 368 / 182 | 0.774 | 0.916 | 0.905 |
| Palbociclib | 370 / 193 | 0.561 | 0.726 | 0.722 |
| Olaparib | 374 / 188 | 0.493 | 0.575 | 0.606 |
| Gemcitabine | 298 / 148 | 0.592 | 0.609 | 0.506 |

The paired XGBoost-minus-linear AUROC differences are −0.016 for trametinib (95% conditional bootstrap interval −0.067 to 0.036) and −0.011 for afatinib (−0.038 to 0.016). These data do not demonstrate added value over the linear baseline. Intervals resample held-out cell-line scores conditional on the fitted folds; they exclude training and fold-selection uncertainty.

**Gemcitabine is the clearest observed cost of complexity:** XGBoost AUROC is 0.506 versus 0.609 for ElasticNet; the difference is **−0.1028**, with 95% paired conditional interval **[−0.1779, −0.0224]**. This unadjusted exploratory interval excludes zero. It is not a multiplicity-adjusted conclusion across all drugs and omits retraining uncertainty. It should be shown openly, not hidden behind the stronger examples.

## Interval-policy feasibility, not validated abstention

The rule accepts **0/375 trametinib** and **0/368 afatinib** cases before release and OOD gates. On these saved predictions the interval/probability rule is inoperative; zero acceptance does not validate its reliability. Selective classification error is undefined. The policy is retained as an auditable prototype restriction; continuous ranking is the usable candidate-selection output.

For a symmetric radius q, a sensitive proposal needs `prediction <= sensitive_max - q`; a resistant proposal needs `prediction >= resistant_min + q`. The class-band width alone does not determine feasibility: response tails are half-lines. A perfect model with smaller calibration residuals could accept cases. In the current palbociclib OOF data, both XGBoost and ElasticNet already have one technically eligible case, disproving a blanket impossibility across all drugs.

The new `interval_feasibility.csv` reports radius, thresholds, predicted range and headroom **per model family and fold**. Do not mix pooled averages or tuned-bundle radii with matched OOF predictions. XGBoost's OOF radii range from **0.2463–0.2911** for trametinib and **0.1768–0.2554** for afatinib. Their saved tuned-bundle radii are different (**0.3318 / 0.3045**).

Holding predictions, class cutoffs and the probability-margin gate fixed, a common multiplier of the fold radii first permits at least one technical decision at **0.5431** for trametinib and **0.8034** for afatinib: reductions of **45.7% / 19.7%** relative to the original radii. These are **radius multipliers, not nominal coverage levels**, not recommended operating points and not acceptance by the full release policy. They use evaluation predictions post hoc and need independent validation.

The original fold calibration residual distributions were not saved. A single 90% quantile cannot identify the quantile at which acceptance starts, so proposed **65–69% / 77–79% nominal coverage values remain unverified**. Future training and OOF runs now preserve calibration residuals; `maximum_nominal_coverage` applies the exact finite-sample rank to a supplied radius limit. No old model or calibration residual has been fabricated or retrained for this audit.

Residual error combines measurement noise, model error and heterogeneity. This analysis does **not** identify an assay-noise ceiling. Marginal interval miscoverage is also not the conditional error among selected categorical decisions.

## Boundary ties and fold allocation

The saved lineage top-ten list uses the documented ascending-ModelID tie break and contains **5** sensitive models. Four of five tied models occupy its boundary: exhaustive tie possibilities give **5–6** hits, with a uniform-tie expected value **5.2**. XGBoost and ElasticNet each have seven with no top-ten boundary ambiguity. Tables now retain the deterministic count and add min/max/expected counts; figures show the expectation and tie range. Outcome labels evaluate those possibilities, never choose the tie break.

Outer and inner fold allocation uses full-cohort response-rank quartiles. This outcome-aware stratification is disclosed in the manifests. Feature selection, class cutoffs and calibration remain local to their proper data roles; the design is not outcome-blind prospective allocation.

## External transfer

Corrected bundles were re-evaluated on GDSC2 for internal test IDs only. Trametinib AUROC is 0.820 (20 extremes among 39 overlaps) and afatinib 0.860 (22 among 43). The previously reported GDSC2 results had already been inspected before this revision, so this is a **retrospective re-evaluation**, not fresh independent confirmation. GDSC2 defines its own response quartiles: AUROC assesses ranking transfer, not transport of the PRISM probability calibration or conformal coverage.

The new bundles in `models/review/` remain `unreviewed`. Historical bundles and their release labels remain in `models/real/`; the new API abstains on them because independent calibration under this protocol is not verified.

## Reproduction

Place the source files named in `reports/pierre_fabre_review/input_manifest.json` in `data/raw/recovered/` for the complete command sequence below. Large original data are excluded from Git. The ZIP hashes identify the recovered Drive copies; the PRISM CSV hash matches the historical manifest.

```bash
python -m pip install -e '.[dev]'
python scripts/run_pierre_fabre_review.py --raw data/raw/recovered --train
python scripts/analyze_screening_budget.py
python scripts/review_external_transfer.py
pytest
```

`review_external_transfer.py` currently uses the review cache at `data/raw/recovered/aligned_review_cache.joblib`; supply the same path when reproducing that last script. The cache is accepted only when source hashes match during preparation. All scripts resolve report locations relative to the repository, except the configurable primary review output.

Source tables, fold memberships, all OOF predictions, paired comparisons, budget rankings, external reanalysis and package versions are in `reports/pierre_fabre_review/`. No Stack GPU experiment or new wet-lab validation was performed in this review.

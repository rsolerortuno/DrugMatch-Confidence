# DrugMatch-Confidence
## Goal map, milestones and autonomous development specification

**Purpose:** Build a GitHub-ready portfolio tool that predicts drug response in cancer cell lines from baseline molecular profiles and reports how trustworthy each prediction is.

**Primary model:** XGBoost  
**Secondary model:** Small PyTorch MLP, only after version 0.1 is complete  
**Training domain:** Preclinical cancer cell lines  
**Training data:** Public PRISM drug-response data plus public DepMap molecular profiles  
**External validation:** Public GDSC2 data  
**Clinical limitation:** The project must never present cell-line predictions as patient treatment recommendations.

---

# 1. Product in one sentence

> Given a cancer model and a selected drug, predict whether the model is likely to be sensitive or resistant, estimate the continuous response, explain the molecular drivers and warn when the prediction is unreliable.

## Inputs

- Gene expression.
- Selected mutations.
- Selected copy-number features.
- Cancer lineage.
- Selected drug.

## Outputs

- Predicted continuous response.
- Sensitive/intermediate/resistant class.
- Calibrated probability.
- Prediction interval.
- Out-of-distribution status.
- SHAP explanation.
- Model, data and feature versions.
- Explicit preclinical-use disclaimer.

---

# 2. North-star goal

The project is complete when a new user can:

1. Clone the repository.
2. Install it with one documented command.
3. Download or prepare the public data reproducibly.
4. Recreate the processed datasets and fixed splits.
5. Train the baselines and XGBoost models.
6. Reproduce internal and external validation.
7. Generate SHAP explanations.
8. Run predictions through a Python API, CLI and Streamlit app.
9. Run the test suite and build the Docker image.
10. Understand the limitations from the README and model card.

## Version 0.1 acceptance criteria

- Five drugs are modelled internally.
- At least two drugs are tested in GDSC2 without tuning on GDSC2.
- XGBoost is compared with naive, lineage-only and Elastic Net baselines.
- Feature selection and preprocessing are fitted using training data only.
- Splits are grouped by cell line.
- Classification probabilities are calibrated.
- Regression outputs include evaluated prediction intervals.
- Out-of-distribution samples are flagged.
- Global and per-sample SHAP explanations work.
- Offline CI runs using synthetic fixtures.
- Large public datasets are not committed to Git.
- Streamlit, CLI, tests, Docker, README and model card are complete.

---

# 3. Scope controls

## Included in v0.1

- Public PRISM response data.
- Public DepMap expression, mutation, copy-number and lineage data.
- Public GDSC2 response data.
- Five data-rich drugs selected through an automated audit.
- Regression and classification.
- Naive, lineage-only, Elastic Net and XGBoost models.
- Calibration, uncertainty and simple OOD detection.
- Leave-one-lineage-out evaluation.
- External validation.
- SHAP.
- API, CLI and Streamlit.
- Tests, CI, Docker and documentation.

## Excluded from v0.1

- Patient response.
- Clinical recommendations.
- Drug combinations.
- Single-cell or spatial data.
- Molecular structures and unseen-drug prediction.
- Transformers or foundation models.
- Reinforcement learning or generative models.
- PyTorch before the complete v0.1 release.

These exclusions are mandatory. They prevent scope creep.

---

# 4. Goal map

```text
G0 Complete GitHub portfolio product
│
├── G1 Reproducible project foundation
│   ├── M01 Repository scaffold
│   ├── M02 Configuration and seeds
│   └── M03 Synthetic fixtures
│
├── G2 Reliable public-data layer
│   ├── M04 Data download and manifests
│   ├── M05 Dataset audit
│   ├── M06 Select five drugs
│   └── M07 Harmonize identifiers
│
├── G3 Leakage-safe modelling dataset
│   ├── M08 Define labels
│   ├── M09 Engineer features
│   ├── M10 Create fixed grouped splits
│   └── M11 Validate processed-data contracts
│
├── G4 Baseline and XGBoost models
│   ├── M12 Naive and lineage baselines
│   ├── M13 Elastic Net
│   ├── M14 XGBoost regression
│   └── M15 XGBoost classification
│
├── G5 Confidence and transferability
│   ├── M16 Probability calibration
│   ├── M17 Regression intervals
│   ├── M18 OOD detection
│   ├── M19 Leave-one-lineage-out validation
│   └── M20 GDSC2 external validation
│
├── G6 Biological interpretation
│   ├── M21 Global SHAP
│   ├── M22 Local SHAP
│   ├── M23 Feature stability
│   └── M24 Modality ablations
│
├── G7 User-facing product
│   ├── M25 Python API
│   ├── M26 CLI
│   ├── M27 Streamlit app
│   └── M28 Example reports
│
├── G8 Engineering and release
│   ├── M29 Tests
│   ├── M30 GitHub Actions
│   ├── M31 Docker
│   ├── M32 README and model card
│   └── M33 Release v0.1
│
└── G9 Optional extension
    ├── M34 Multi-drug PyTorch dataset
    ├── M35 Small shared MLP
    ├── M36 XGBoost-versus-MLP benchmark
    └── M37 Release v0.2
```

---

# 5. Milestones and concrete definitions of done

## M01 — Repository scaffold

Create:

```text
DrugMatch-Confidence/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── .pre-commit-config.yaml
├── configs/
├── data/{raw,interim,processed}/
├── notebooks/
├── src/drugmatch/
├── app/
├── tests/{unit,integration,fixtures}/
├── models/
├── reports/
├── scripts/
├── Dockerfile
└── .github/workflows/
```

**Done when:**

- `pip install -e .` succeeds.
- `python -m drugmatch --help` succeeds.
- `pytest` runs.
- Large data and model artifacts are ignored by Git.

---

## M02 — Configuration and reproducibility

Create:

- `configs/project.yaml`
- `configs/data_sources.yaml`
- `configs/features.yaml`
- `configs/models/elastic_net.yaml`
- `configs/models/xgboost.yaml`
- `configs/evaluation.yaml`

Store:

- public release identifiers;
- random seed;
- drug-selection thresholds;
- response thresholds;
- feature limits;
- split proportions;
- search spaces;
- output paths.

**Done when:**

- No important modelling constant is hidden in a notebook.
- One global seed controls NumPy, scikit-learn and XGBoost.
- The CLI can display the resolved configuration.

---

## M03 — Synthetic fixtures

Create miniature expression, mutation, copy-number, metadata, response and external-response datasets.

Include intentional:

- missing values;
- duplicate IDs;
- invalid drug;
- ambiguous mapping;
- class imbalance.

**Done when:**

- The complete miniature pipeline runs offline on CPU.
- CI never needs to download real data.

---

## M04 — Data download and manifests

Implement:

```bash
drugmatch data download --source depmap
drugmatch data download --source prism
drugmatch data download --source gdsc2
```

Record:

- source;
- release;
- URL or download identifier;
- checksum when available;
- local file;
- date;
- schema version.

**Done when:**

- Downloads are resumable and idempotent.
- `--force` is supported.
- Schema changes fail clearly.
- Raw data remain outside Git.

---

## M05 — Dataset audit

For every candidate drug, calculate:

- number of usable cell lines;
- missingness;
- response distribution and variance;
- sensitive and resistant counts;
- lineage diversity;
- molecular-data overlap;
- GDSC2 overlap;
- ambiguity and rejection reason.

Outputs:

- `reports/data_audit/drug_audit.csv`
- `reports/data_audit/data_audit.md`

**Done when:**

- The report can be generated without training any model.
- Every rejected drug has an explicit reason.

---

## M06 — Select five drugs

Default criteria:

- at least 350 usable PRISM models;
- no more than 20% missing response;
- at least 50 sensitive and 50 resistant models;
- at least five cancer lineages;
- adequate response variability;
- sufficient molecular coverage;
- preference for GDSC2 overlap.

Desired mix:

- two targeted therapies;
- one drug with a known biomarker;
- one heterogeneous-response drug;
- one mechanistically distinct drug.

Output:

- `configs/selected_drugs.yaml`

**Decision gate DG1:** Do not continue if five drugs do not pass the audit or fewer than two support credible external validation.

---

## M07 — Harmonize identifiers

Create deterministic mapping tables for:

- cell-line identifiers;
- drug identifiers;
- original source identifiers;
- aliases;
- mapping provenance.

Never silently resolve ambiguous matches.

Outputs:

- `data/interim/model_mapping.parquet`
- `data/interim/drug_mapping.parquet`
- unmatched and ambiguous reports.

**Done when:**

- Mapping coverage and exclusions are reported.
- Duplicate and many-to-one cases are tested.
- Manual overrides are versioned and justified.

---

## M08 — Define labels

### Regression

Preserve the original continuous PRISM response and calculate a within-drug standardized or percentile response.

### Classification

- lower 25%: sensitive;
- upper 25%: resistant;
- central 50%: intermediate;
- first binary model uses sensitive versus resistant.

Thresholds must be configurable.

**Done when:**

- Class counts are reported.
- No GDSC2 response is used to create PRISM labels.

---

## M09 — Engineer features

v0.1 feature groups:

- expression;
- recurrent mutations;
- selected copy-number features;
- lineage.

Default compact representation:

- 300 variable expression genes;
- recurrent mutations above a configured prevalence;
- limited copy-number features;
- one-hot lineage.

**Critical rule:** All fit-dependent preprocessing and feature selection occur inside a pipeline fitted on training data only.

**Done when:**

- Feature names and manifests are preserved.
- Missing-value behaviour is explicit.
- Leakage tests pass.

---

## M10 — Fixed grouped splits

Create:

- 70% train;
- 15% validation;
- 15% internal test;
- grouping by cell line.

For multi-drug tables, every row from the same cell line stays in one partition.

Also define leave-one-lineage-out folds.

Output:

- `data/processed/splits.json`

**Done when:**

- No model ID crosses partitions.
- Class and lineage balance are reported.
- All algorithms reuse the same immutable splits.
- Test data remain untouched until final evaluation.

---

## M11 — Processed-data contracts

Define validated objects such as:

- `ModelFeatureMatrix`
- `DrugResponseLabels`
- `DatasetSplit`
- `FeatureManifest`
- `DataProvenance`

**Done when:**

- Invalid shapes, duplicated rows and missing IDs fail early.
- Processed datasets contain hashes and provenance.
- Training never reads raw files directly.

**Decision gate DG2:** Do not tune models until contracts, split-integrity and leakage tests pass.

---

## M12 — Naive and lineage baselines

Regression:

- global mean;
- per-lineage mean.

Classification:

- majority class;
- lineage-only logistic regression.

**Done when:**

- Baseline metrics exist for all five drugs.
- They use the same stored splits as later models.

---

## M13 — Elastic Net

Implement classification and regression where appropriate.

**Done when:**

- Hyperparameters are selected without the internal test set.
- Non-zero coefficients and selected features are saved.
- Results are compared with naive and lineage baselines.

---

## M14 — XGBoost regression

Predict continuous response for each selected drug.

Tune only a compact search space:

- trees;
- learning rate;
- maximum depth;
- row and column subsampling;
- regularization;
- minimum child weight.

**Done when:**

- One saved model per drug.
- Early stopping is supported.
- Model, configuration, preprocessing and feature manifest are bundled.
- Internal-test predictions are saved once and treated as immutable.

---

## M15 — XGBoost classification

Predict sensitive versus resistant.

Report:

- AUROC;
- AUPRC;
- balanced accuracy;
- precision;
- recall;
- MCC;
- Brier score;
- confusion matrix.

**Done when:**

- Class imbalance is handled explicitly.
- Raw probability outputs are preserved for calibration.

**Decision gate DG3:** Continue even with mixed results, but report honestly whether molecular models outperform lineage-only baselines.

---

## M16 — Probability calibration

Compare:

- Platt scaling;
- isotonic regression.

Select using validation data only.

**Done when:**

- Calibration curves and Brier scores are shown before and after calibration.
- Calibration objects are packaged with classification models.

---

## M17 — Regression prediction intervals

Use split conformal prediction or another simple transparent method.

**Done when:**

- Every prediction has lower and upper bounds.
- Empirical coverage and interval width are reported.
- Target coverage is configurable.

---

## M18 — Out-of-distribution detection

MVP method:

1. Apply fitted preprocessing.
2. Create a PCA representation using training data.
3. Measure distance or density.
4. Define in-distribution, caution and OOD thresholds.

**Done when:**

- OOD fitting uses training data only.
- Held-out lineages are used as a stress test.
- API, CLI and app expose the warning.
- OOD is described as a warning, not proof of error.

---

## M19 — Leave-one-lineage-out validation

For at least four well-represented lineages:

- remove the lineage from all fitting;
- train;
- evaluate only on that lineage;
- compare with random-split performance;
- examine OOD scores.

**Done when:**

- Transfer failure is visible rather than hidden.
- Results are stored per lineage and drug.

---

## M20 — GDSC2 external validation

Rules:

- train only on PRISM;
- freeze preprocessing and model;
- never tune using GDSC2 outcomes;
- match exact drugs and compatible models;
- compare ranks or standardized responses rather than assuming raw scales are identical.

Report:

- overlap;
- Spearman correlation;
- sensitive/resistant performance;
- top-sensitive-model recovery;
- failure cases.

**Done when:**

- At least two drugs are evaluated externally.
- Inclusion and exclusion rules are documented.
- Dataset differences are discussed.

**Decision gate DG4:** Moderate external performance is acceptable if evaluation is correct, reproducible and honest.

---

## M21 — Global SHAP

Generate:

- summary plots;
- feature-importance tables;
- feature groups by modality.

**Done when:**

- Explanations use the correct transformed feature names.
- Results are created for all final drug models.

---

## M22 — Local SHAP

For each sample return:

- top sensitivity drivers;
- top resistance drivers;
- confidence;
- interval;
- OOD warning.

Avoid causal language.

**Done when:**

- CLI and Streamlit generate reproducible local explanations.

---

## M23 — Feature stability

Measure:

- top-feature overlap between folds;
- rank correlations;
- directional consistency;
- performance without lineage;
- agreement with Elastic Net.

**Done when:**

- Stable and unstable features are explicitly labelled.
- New biomarkers are presented only as hypotheses.

---

## M24 — Modality ablations

Compare:

- lineage only;
- expression only;
- expression + lineage;
- mutation + copy number;
- all modalities;
- all modalities without lineage.

**Done when:**

- Identical splits and comparable tuning budgets are used.
- The smallest competitive model is identified.

---

## M25 — Python API

Target interface:

```python
from drugmatch import DrugMatchPredictor

predictor = DrugMatchPredictor.load("drug_name")
result = predictor.predict(
    expression=expression,
    mutations=mutations,
    copy_number=copy_number,
    lineage="lung",
)
```

Result must include:

- response;
- interval;
- class;
- calibrated probability;
- confidence;
- OOD;
- top features;
- versions;
- disclaimer.

---

## M26 — CLI

Required commands:

```bash
drugmatch data download
drugmatch data audit
drugmatch data build
drugmatch train baseline
drugmatch train xgboost
drugmatch evaluate internal
drugmatch evaluate external
drugmatch explain
drugmatch predict
drugmatch app
```

**Done when:**

- Every command has `--help`.
- Errors and exit codes are meaningful.
- Logs are saved.

---

## M27 — Streamlit app

### Demo mode

- select a drug;
- select a known DepMap model;
- display prediction.

### Upload mode

- upload documented feature template;
- validate compatibility;
- return prediction or a clear error.

Panels:

- response;
- confidence;
- interval;
- OOD;
- SHAP drivers;
- similar training examples;
- preclinical disclaimer.

---

## M28 — Example reports

Create examples for:

- confident sensitive;
- confident resistant;
- uncertain;
- OOD;
- external-validation case.

Each example states what can and cannot be concluded.

---

## M29 — Tests

Unit tests:

- schemas;
- mappings;
- feature selection;
- leakage;
- split integrity;
- calibration;
- intervals;
- OOD;
- serialization;
- API.

Integration tests:

- synthetic data build;
- training;
- evaluation;
- prediction;
- CLI;
- app model loading.

**Done when:**

- All tests run offline.
- Critical pipeline outputs have regression tests.

---

## M30 — GitHub Actions

Jobs:

- lint;
- type checking;
- unit tests;
- integration tests;
- package build;
- optional Docker build.

---

## M31 — Docker

**Done when:**

- CLI and Streamlit run in the image.
- Real data and trained models are mounted as volumes.
- Raw public data are not embedded in the image.

---

## M32 — README and model card

README:

1. Question.
2. Data.
3. Workflow.
4. Installation.
5. Quick start.
6. Reproduction.
7. Results.
8. Limitations.
9. Structure.
10. Roadmap.

Model card:

- intended and excluded use;
- training and evaluation data;
- metrics;
- calibration;
- external validation;
- OOD;
- limitations;
- clinical warning;
- versioning.

---

## M33 — Release v0.1

Artifacts:

- tagged release;
- changelog;
- source;
- configurations;
- data manifests;
- reports;
- model card;
- demo models or documented model download;
- application screenshots.

**Final definition of done:** A reviewer can understand, reproduce and run the project without reading the development chat.

---

# 6. Optional PyTorch extension

This section is blocked until M33 is complete.

## M34 — Multi-drug dataset

Pool 20–30 sufficiently measured drugs while preserving grouped cell-line splits.

## M35 — Small shared MLP

Example:

```text
molecular features
      ↓
Dense 256 + ReLU + dropout
      ↓
Dense 64
      +
drug embedding
      ↓
response head
```

Requirements:

- plain PyTorch;
- explicit training loop;
- early stopping;
- checkpointing;
- deterministic seeds;
- one ordinary GPU is sufficient;
- synthetic CPU test.

## M36 — Compare XGBoost and MLP

Compare:

- predictive performance;
- calibration;
- external transfer;
- compute;
- data-size sensitivity;
- interpretability.

There is no requirement that deep learning wins.

## M37 — Release v0.2

Release only if the MLP provides meaningful capability or a useful technical comparison.

---

# 7. GitHub issue map

| Issue | Title | Dependencies |
|---:|---|---|
| 001 | Scaffold package and repository | — |
| 002 | Add configuration and reproducibility | 001 |
| 003 | Add synthetic fixtures | 001 |
| 004 | Implement data download and manifests | 002 |
| 005 | Implement dataset audit | 003, 004 |
| 006 | Implement automatic drug selection | 005 |
| 007 | Harmonize model and drug identifiers | 004 |
| 008 | Define regression and classification labels | 006, 007 |
| 009 | Implement leakage-safe preprocessing | 003, 007 |
| 010 | Create fixed grouped splits | 008, 009 |
| 011 | Add processed-data contracts | 008–010 |
| 012 | Add naive and lineage baselines | 011 |
| 013 | Add Elastic Net | 012 |
| 014 | Add XGBoost regression | 013 |
| 015 | Add XGBoost classification | 013 |
| 016 | Add probability calibration | 015 |
| 017 | Add regression intervals | 014 |
| 018 | Add OOD scoring | 014, 015 |
| 019 | Build internal evaluation reports | 014–018 |
| 020 | Add leave-one-lineage-out evaluation | 019 |
| 021 | Add GDSC2 external validation | 007, 019 |
| 022 | Add global and local SHAP | 014, 015 |
| 023 | Add feature-stability analysis | 022 |
| 024 | Add modality ablations | 019 |
| 025 | Implement prediction API | 016–018, 022 |
| 026 | Implement CLI | 025 |
| 027 | Build Streamlit app | 025 |
| 028 | Generate example reports | 021–027 |
| 029 | Complete unit and integration tests | continuous |
| 030 | Add GitHub Actions | 029 |
| 031 | Add Docker | 026, 027 |
| 032 | Complete README and model card | 028 |
| 033 | Tag v0.1 | 030–032 |
| 034 | Build multi-drug PyTorch dataset | 033 |
| 035 | Implement shared MLP | 034 |
| 036 | Benchmark XGBoost versus MLP | 035 |
| 037 | Tag v0.2 | 036 |

---

# 8. Rules for Autopilot

## Scope

1. Work on one issue at a time.
2. Do not start PyTorch before v0.1.
3. Do not add new modalities outside an accepted issue.
4. Do not refactor unrelated modules.
5. Do not change contracts without tests and documentation.
6. Never tune on the internal test set or GDSC2.
7. Never make patient-level claims.

## Code

1. Production logic belongs in `src/drugmatch/`.
2. Notebooks are exploratory only.
3. Public functions use type hints and concise docstrings.
4. Validate schemas, IDs and paths.
5. Fail on ambiguous mappings.
6. Save provenance with datasets and models.
7. Pin dependencies.
8. Use deterministic seeds where possible.
9. Avoid unnecessary dependencies.
10. CI uses synthetic fixtures.

## Git

1. One branch per issue: `feature/NNN-short-name`.
2. Keep commits focused.
3. Never commit raw PRISM, DepMap or GDSC files.
4. Never commit secrets.
5. Update the changelog for visible changes.
6. Open a draft PR after tests pass.
7. Include acceptance evidence in the PR.

## Issue completion gate

An issue is complete only when:

- requested behaviour exists;
- success and failure tests exist;
- docs are updated;
- lint, types and tests pass;
- no unrelated changes are included;
- acceptance criteria are demonstrated.

---

# 9. Required final outputs

```text
reports/
├── data_audit/
├── internal_validation/
│   ├── model_comparison.csv
│   ├── regression_metrics.csv
│   ├── classification_metrics.csv
│   ├── calibration/
│   └── lineage_holdout/
├── external_validation/
│   ├── gdsc2_overlap.csv
│   ├── gdsc2_metrics.csv
│   └── external_validation_report.html
├── interpretation/
│   ├── shap_global/
│   ├── shap_local/
│   ├── feature_stability.csv
│   └── modality_ablation.csv
├── examples/
└── model_card.md
```

Minimum final figures:

1. Workflow.
2. Data coverage.
3. Baseline comparison.
4. Regression performance.
5. Classification performance.
6. Calibration.
7. Interval coverage.
8. Leave-one-lineage-out performance.
9. External validation.
10. Global SHAP.
11. Local SHAP.
12. Modality ablation.

---

# 10. Portfolio wording

## One sentence

> DrugMatch-Confidence is an interpretable and uncertainty-aware machine-learning framework that predicts preclinical cancer-model drug response from baseline molecular profiles and evaluates whether each prediction is trustworthy.

## CV version

> Developed a reproducible pharmacogenomic ML framework using public DepMap, PRISM and GDSC resources. Compared lineage baselines, Elastic Net, XGBoost and an optional PyTorch model; implemented leakage-safe evaluation, external transfer testing, calibration, prediction intervals, OOD detection, SHAP, CLI, Streamlit, Docker and CI.

---

# 11. Copy-paste prompt for the development chat

## BEGIN DEVELOPMENT PROMPT

Develop a GitHub-ready project named **DrugMatch-Confidence**.

The goal is to build an interpretable and uncertainty-aware machine-learning tool that predicts preclinical cancer cell-line response to selected drugs from baseline molecular profiles.

### Core output

Given expression, selected mutations, selected copy-number features, lineage and a selected drug, return:

1. continuous predicted response;
2. sensitive/intermediate/resistant class;
3. calibrated confidence;
4. prediction interval;
5. out-of-distribution status;
6. SHAP explanation;
7. clear preclinical-use disclaimer.

### Data

Use public and reproducibly downloadable:

- PRISM drug-response data for training;
- DepMap molecular profiles and model metadata;
- GDSC2 for independent external validation.

Pin exact releases in configuration. Do not commit large raw datasets. Create data manifests with source, release, file, checksum and provenance.

### Required v0.1 scope

- five drugs selected automatically using data-quality criteria;
- at least two drugs with usable GDSC2 validation;
- expression, mutations, copy number and lineage;
- naive, lineage-only and Elastic Net baselines;
- XGBoost regression and classification;
- probability calibration;
- transparent regression intervals;
- simple PCA/distance-based OOD detection;
- leave-one-lineage-out evaluation;
- GDSC2 validation without tuning;
- SHAP and modality ablations;
- Python API;
- CLI;
- Streamlit;
- offline tests using synthetic fixtures;
- GitHub Actions;
- Docker;
- README and model card.

### Strict v0.1 exclusions

Do not add patient-response claims, drug combinations, single-cell data, spatial data, chemical graph encoders, unseen-drug prediction, transformers, foundation models, reinforcement learning, generative models or PyTorch before the complete v0.1 release.

### Engineering rules

- Work issue by issue.
- Production code belongs in `src/drugmatch/`.
- Notebooks are not the source of truth.
- Use typed and validated contracts.
- Fit preprocessing and feature selection using training data only.
- Group all splits by cell line.
- Never tune on the internal test set or GDSC2.
- Use synthetic fixtures so CI runs offline.
- Never silently resolve ambiguous IDs.
- Save configuration, manifests, hashes, features and provenance with models.
- Keep results deterministic where possible.
- Write tests before declaring an issue complete.
- Avoid unrelated refactors.
- Never commit datasets, secrets or tokens.

### Execution order

1. Scaffold repository.
2. Add configuration and synthetic fixtures.
3. Implement downloads and manifests.
4. Audit data.
5. Select five drugs.
6. Harmonize IDs.
7. Define labels.
8. Build leakage-safe features.
9. Create fixed grouped splits.
10. Add processed-data contracts.
11. Train baselines and Elastic Net.
12. Train XGBoost regression and classification.
13. Add calibration, intervals and OOD.
14. Run internal and leave-one-lineage-out evaluation.
15. Run GDSC2 external validation.
16. Add SHAP, stability and modality ablations.
17. Build API, CLI and Streamlit.
18. Complete tests, CI, Docker, README and model card.
19. Tag v0.1.
20. Only then consider a small PyTorch MLP for v0.2.

### Definition of done

v0.1 is complete only when five drugs are evaluated internally; at least two are evaluated externally; XGBoost is compared with all baselines; probabilities are calibrated; intervals and OOD warnings work; SHAP works globally and locally; offline CI passes; the real-data workflow is reproducible; Streamlit and Docker work; and all documentation clearly states that predictions concern preclinical cell lines, not patients.

Start by inspecting the current repository in read-only mode. Then propose and implement **Issue 001 only**. Do not start later milestones until the current issue passes its acceptance criteria.

## END DEVELOPMENT PROMPT

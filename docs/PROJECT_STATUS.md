# Project status

## Version 1.0.0

Complete:

- real DepMap Public 26Q1 molecular integration;
- real PRISM training for five drugs;
- XGBoost regression and classification;
- naive, lineage and Elastic Net baselines;
- calibration, conformal intervals and OOD detection;
- fixed test and five-fold OOF validation;
- frozen GDSC2 external validation;
- SHAP, feature stability, lineage holdout and modality ablation;
- Python API, CLI and Streamlit app;
- offline tests, CI configuration, Docker, wheel and release ZIP;
- non-expert README and data documentation.

Model release status:

- trametinib: validated demo;
- afatinib: validated demo;
- palbociclib: exploratory;
- olaparib: insufficient evidence;
- gemcitabine: insufficient evidence.

Future work should improve patient relevance, add better pathway representations, evaluate organoids and preserve the current frozen test/external boundaries.

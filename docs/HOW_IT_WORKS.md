# How DrugMatch-Confidence works

## 1. Select one drug

Each model is drug-specific. Version 1.0 supports trametinib, afatinib, palbociclib, olaparib and gemcitabine.

## 2. Match molecular data to measured PRISM responses

All tables are joined using stable DepMap `ModelID` values. Only models with the required molecular measurements and a valid PRISM response for the selected drug are retained.

## 3. Lock the data split

Models are divided into training, validation and internal test groups. A model appears in only one group. The continuous response distribution is stratified to preserve sensitive and resistant examples.

## 4. Define sensitive and resistant classes without test leakage

The lower and upper PRISM AUC quartiles are calculated from the training group only. Lower-AUC models are sensitive; higher-AUC models are resistant. Middle-quartile models remain useful for regression but are not used in the binary classifier.

## 5. Select molecular variables using training data only

The original joined matrix has 58,498 variables. Within each training fold, the pipeline selects a compact set of variable expression genes, recurrent mutation features, variable copy-number features, global signatures and a small mechanism-informed set. Test samples never influence this selection.

## 6. Train baselines

The project first measures how far simple approaches can go:

- global or per-lineage mean response;
- lineage-only classifier;
- Elastic Net regression and classification.

XGBoost is useful only if it adds value beyond these controls.

## 7. Train XGBoost

A regression model predicts continuous drug-response AUC. A classification model predicts the sensitive class. A compact validation search chooses hyperparameters; the locked test group is not used for tuning.

## 8. Make probabilities and intervals honest

- Platt or isotonic calibration converts raw classifier scores into probabilities.
- The decision threshold is selected on validation data.
- Split-conformal prediction adds a response interval.
- PCA-distance OOD detection warns about unfamiliar samples.
- Input feature coverage lowers confidence when many expected values are absent.

## 9. Explain predictions

SHAP values identify features that push an individual prediction toward sensitivity or resistance. Global SHAP rankings and cross-fold stability are also reported. These values describe model associations, not causal proof.

## 10. Validate in three ways

1. A fixed internal test set.
2. Five-fold out-of-fold predictions, where every model is predicted by a fold that did not train on it.
3. Frozen transfer to GDSC2, including a strict subset restricted to PRISM test IDs.

## 11. Assign a release status

A strong internal result alone is not enough. The release label combines out-of-fold performance and strict external transfer:

- `validated_demo`: suitable as the main portfolio demonstration;
- `exploratory`: useful signal but unstable transfer;
- `insufficient_evidence`: retained as an honest weak or negative result.

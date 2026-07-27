# Biological rationale

Drug response is determined by more than mutation of the nominal drug target. The project therefore combines complementary molecular views:

- expression captures pathway activity and cell state;
- hotspot mutations capture activating oncogenes;
- damaging mutations capture loss of tumour suppressors and repair genes;
- copy number captures amplifications and deletions;
- global signatures capture broader genome instability;
- lineage represents tissue context and is also evaluated as an explicit baseline.

The five drugs were chosen to span distinct mechanisms: MAPK inhibition, ERBB inhibition, cell-cycle inhibition, PARP inhibition and a nucleoside analogue. This diversity reveals where a compact tabular model works and where the available molecular representation is insufficient.

The project does not interpret every high-SHAP feature as a biomarker. Biological confidence increases when a feature is mechanistically plausible, stable across folds, adds performance beyond lineage and transfers to an external dataset.

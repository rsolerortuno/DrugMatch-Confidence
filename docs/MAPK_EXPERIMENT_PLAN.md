# MAPK case: a ten-model experiment to test prioritization

In the retrospective matched OOF evaluation of 375 models, the ten lowest predicted trametinib AUCs contain **7 sensitive models** for both XGBoost and the linear model, versus **5–6** for lineage depending on the boundary tie break (**5.2 expected** under uniform tie breaking; the saved ModelID-ordered list gives 5). Uniform random selection would average **2.53** because 95/375 belong to the training-defined sensitive tail. The observed **2.76-fold enrichment** is a descriptive result from ten selections, not a statistically established advantage over the linear baseline. At budget 20, linear selects 15 sensitive models and XGBoost 13.

Candidate CSVs contain stable ModelIDs, lineage, scores, intervals and retrospective outcomes. Outcomes score the lists; they do not determine their membership. These demonstrate a procedure, not untested prospective recommendations.

## Relevance to Pierre Fabre

MAPK connects this example to the company's oncology portfolio, including BRAF inhibitor encorafenib and MEK inhibitor binimetinib ([official products](https://www.pierre-fabre.com/en/oncology/our-products), reviewed 5 September 2026). Trametinib performance cannot be transferred to those compounds without compound-specific labels and validation. The reusable deliverable is the selection workflow.

## Proposed pilot, not yet executed

1. Freeze compound, exposure endpoint, assay protocol, eligible models, exclusions and selection rule before new outcomes. Use independent new model/assay data and auditable identities.
2. Retain matched linear and XGBoost comparators. Add a mechanism-prior comparator defined by project biologists, such as an exact BRAF mutation/lineage rule, frozen before testing. This comparator was not evaluated in this revision.
3. Allocate ten model slots: six sensitivity-ranked candidates, two predicted resistant controls, and two discordant or uncertain cases. Assay controls and independent replicate wells are additional: ten models does not mean ten wells.
4. Measure viability dose-response and pathway target engagement, for example phospho-ERK, with independent biological replicates and predefined assay QC. A viability hit without pathway engagement triggers a mechanism review. Retain negative results.
5. Primary readout: enrichment among the six ranked candidates against the predefined reference. Secondary: continuous rank association, replicate agreement, and responses of the two resistant and two uncertain models. Report exclusions. Do not score the mixed ten-model panel as pure top-10 precision.
6. Add gene-dependency evidence as an orthogonal hypothesis check. CRISPR knockout, CRISPRi and pharmacological inhibition have distinct mechanisms and endpoints.

The small pilot tests assay feasibility and prioritization, not definitive biomarker validity. The current categorical policy is inoperative for trametinib on the saved OOF panel. A post-hoc radius-feasibility audit quantifies this limitation; it does not validate a new coverage setting. The pilot evaluates ranking, not this unsupported categorical deployment.

## Connection to Stack

Stack tests transcriptional effect transfer; DrugMatch tests baseline context against measured drug response. Transcriptomic similarity is not target efficacy. These remain separate evidence streams until linked experimentally.

## Interview wording

“I built evaluations that could rule out an unsupported modelling choice before a larger experimental investment. Here they flag an inoperative categorical policy and show that more complex modelling can hurt: for gemcitabine, XGBoost loses about 0.10 AUROC to the linear baseline. For the lead MAPK example, both molecular models recover seven sensitive models in a retrospective top-ten panel, so I would take the simpler comparator into a controlled ranking-and-mechanism pilot. The reusable output is the decision process, the controls and a reproducible shortlist—not a claim that either model is ready for automatic deployment.”

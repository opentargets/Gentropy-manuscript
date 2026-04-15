# Figure-to-Code Mapping

Maps every figure in the manuscript to the analysis code and data files in this
repo.

Paper source: `~/Projects/manuscript_gentropy/` Analysis repo:
`~/Projects/Gentropy-manuscript/` (this repo)

---

## Main Figures

### Figure 1 — Panoramic view across 100,526 complex-trait GWAS

**Paper label:** `fig:1` | **Paper file:** `figures/figure_1.pdf`

| Panel    | Description                                     | Code                                                                           | Data                                                                                                                                                  |
| -------- | ----------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| A        | Circular Manhattan / pleiotropy map             | `chapters/03-manuscript-figures/figure_1/manh_plot_dataprep.ipynb` (data prep) | `chapters/03-manuscript-figures/figure_1/disease_ta_index_pandas.csv`, `figure_1/genes_therapeutic_areas/`, `figure_1/target_index_for_plot.parquet/` |
| B–C      | Temporal trends (sample size, GWAS discoveries) | `chapters/03-manuscript-figures/figure_1/Figure_1_b_c.R`                       | `chapters/03-manuscript-figures/figure_1/data/l2g_diseases_full.csv`, `data/qd_sl_eff.csv`, `data/qm_sl_eff.csv`                                      |
| D        | Gene/disease per therapeutic area               | `chapters/03-manuscript-figures/figure_1/Figure_1_d.R`                         | `chapters/03-manuscript-figures/figure_1/l2g_diseases_full.csv`                                                                                       |
| Combined | Assembly                                        | `chapters/03-manuscript-figures/figure_1/Figure_1_combined.R`                  | (reads outputs of above)                                                                                                                              |

**Upstream analysis:**
`chapters/02-analysis/01-descriptions-numbers/01_descriptive_numbers.ipynb`

---

### Figure 2 — Dependency between variant effect size, MAF and predicted consequence

**Paper label:** `fig:2` | **Paper file:** `figures/figure_2.pdf`

| Panel | Description                                                           | Code                                                 | Data                                                                                                                |
| ----- | --------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| A–D   | Effect size vs MAF, PAV proportion, consequence groups, variant types | `chapters/03-manuscript-figures/figure_2/figure_2.R` | `chapters/03-manuscript-figures/figure_2/data/figure_2_a.csv`, `figure_2_b.csv`, `figure_2_c.csv`, `figure_2_d.csv` |

**Upstream analysis (data preparation):**

- `chapters/02-analysis/02-variant-effects/01_lead_variant_effect_filtering.ipynb`
- `chapters/02-analysis/02-variant-effects/02_variant_functional_consequence.ipynb`
- `chapters/02-analysis/02-variant-effects/03_variant_regulatory_consequence.ipynb`
- `playground/plots/figure_2_dataprep_c_d.ipynb` (panel C/D data prep)

---

### Figure 3 — Variant-level pleiotropy modelling

**Paper label:** `fig:3` | **Paper file:** `figures/figure_3.pdf`

| Panel    | Description                     | Code                                                                                       | Data                                                                                |
| -------- | ------------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| A        | vPS vs MAF (observed/predicted) | `chapters/03-manuscript-figures/figure_3/R_scripts/clustering_plot_a_b.R`                  | `chapters/03-manuscript-figures/figure_3/data/plot_a.csv`                           |
| B        | Forest plot: vPS covariates     | `chapters/03-manuscript-figures/figure_3/R_scripts/clustering_plot_a_b.R`                  | `chapters/03-manuscript-figures/figure_3/data/plot_b.csv`                           |
| C        | APOE scatter plots              | `chapters/03-manuscript-figures/figure_3/python_scripts/variant_pleiotropy_analysis.ipynb` | `chapters/03-manuscript-figures/figure_3/data/variant_pleiotropy_data_exploded.csv` |
| Combined | Assembly                        | `chapters/03-manuscript-figures/figure_3/figure_3.R`                                       | (reads panel outputs)                                                               |

**Upstream analysis (data preparation):**

- `chapters/02-analysis/04-variant-level-ps/01_variant_level_pleiotropy.ipynb`
- `chapters/02-analysis/04-variant-level-ps/02_clustering_analysis.ipynb`
- `chapters/03-manuscript-figures/figure_3/python_scripts/prepare_plot_a_b_data.py`

---

### Figure 4 — Gene-level pleiotropy modelling

**Paper label:** `fig:4` | **Paper file:** `figures/figure_4.pdf`

| Panel    | Description                                  | Code                                                                                                                   | Data                                                                                                                             |
| -------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| A        | Temporal trends gPS/vPS + variants per gene  | `chapters/03-manuscript-figures/figure_4/plot_a.R`                                                                     | `data/intermediate_files/Fig4A_stats_gene_pleiotropy.csv`, `Fig4A_stats_variant_pleiotropy.csv`, `Fig4A_stats_gene_coverage.csv` |
| B        | NB regression forest plot (gPS covariates)   | `chapters/03-manuscript-figures/figure_4/plot_b.R`                                                                     | `data/figure_4/gene_pleiotropy_full_model.csv` (→ `data/intermediate_files/gene_pleiotropy_full_model.csv`)                      |
| C        | Gene set enrichment (log-odds, 21 gene sets) | `chapters/03-manuscript-figures/figure_4/plot_d.R` + `chapters/03-manuscript-figures/figure_4/python_scr/plot_d.ipynb` | `data/figure_4/`                                                                                                                 |
| Combined | Assembly                                     | `chapters/03-manuscript-figures/figure_4/figure_4.R`                                                                   | (reads panel outputs)                                                                                                            |

**Upstream analysis:**

- `chapters/02-analysis/05-gene-level-ps/01_gene_level_pleiotropy.ipynb`
- `chapters/02-analysis/05-gene-level-ps/02_temporal_vPS_gPS.ipynb`
- `chapters/02-analysis/05-gene-level-ps/03_average_gPS_gene_categories.ipynb`
- `chapters/02-analysis/05-gene-level-ps/04_gene_pleiotropy_by categories.ipynb`

---

### Figure 5 — Genetic evidence and therapeutic implications

**Paper label:** `fig:5` | **Paper file:** `figures/figure_5.pdf`

| Panel | Description                                                 | Code                                                 | Data                                                                                    |
| ----- | ----------------------------------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------------- |
| A     | Temporal drug-target enrichment OR + cumulative T-I pairs   | `chapters/03-manuscript-figures/figure_5/figure_5.R` | `data/figure_5/temporal_drug_enrichment_full_chembl.csv` (→ `data/intermediate_files/`) |
| B     | Forest plot: enrichment by pleiotropy/effect/variant bins   | `chapters/03-manuscript-figures/figure_5/figure_5.R` | `data/figure_5/drug_enrichment_subsets_vs_full_l2g.csv`                                 |
| C     | Logistic regression probabilities (therapeutic areas + gPS) | `chapters/03-manuscript-figures/figure_5/figure_5.R` | `data/figure_5/df_for_enrichment_regression.csv`                                        |

**Upstream analysis:**

- `chapters/02-analysis/06-target-enrichment/03_temporal_drug_enrichment.ipynb`
- `chapters/02-analysis/06-target-enrichment/05-regression_framework.ipynb`
- `chapters/02-analysis/06-target-enrichment/07-split_by_TA_target_class.ipynb`
- `chapters/02-analysis/06-target-enrichment/02-enrichment-groups.ipynb`
- `chapters/02-analysis/06-target-enrichment/11-non-linearity-gPS.ipynb`

---

## Extended Data Figures

> Extended figures are generated within analysis notebooks in
> `chapters/02-analysis/`. Currently **no dedicated rendering scripts** exist
> for them (unlike main figures).

### Extended Data Figure 1 — Data flowchart

**Paper label:** `fig:ed1` | **Paper file:** `figures/extended_figure_1.png`

| Description                                                           | Code | Notes                                                                                 |
| --------------------------------------------------------------------- | ---- | ------------------------------------------------------------------------------------- |
| Schematic of data flow from 4 sources through fine-mapping to outputs | —    | Created externally (e.g. illustration tool). Not reproducible from code in this repo. |

---

### Extended Data Figure 2 — Credible sets vs GWAS sample size

**Paper label:** `fig:ed2` | **Paper file:** `figures/extended_figure_2.png`

| Description                                    | Code                                                                        | Data                                                |
| ---------------------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------- |
| Average number of CSs per log10(N_samples) bin | `chapters/02-analysis/01-descriptions-numbers/01_descriptive_numbers.ipynb` | `data/intermediate_files/qualifying_credible_sets/` |

---

### Extended Data Figure 3 — Temporal gene-measurement associations

**Paper label:** `fig:ed3` | **Paper file:** `figures/extended_figure_3.png`

| Description                                                                                                                                                               | Code                                                                                   | Data                                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Two-panel stacked bar: cumulative unique measurement-associated genes (top) and gene–measurement pairs (bottom) by year, stratified by EUR common / Non-EUR common / Rare | `chapters/03-manuscript-figures/extended_figures/ed3_temporal_measurement_genes.ipynb` | `data/intermediate_files/list_of_prioritised_genes_per_CS_with_year_nfe_maf.parquet`, `data/intermediate_files/qualifying_measurement_credible_sets/` |

---

### Extended Data Figure 4 — Effect size for eQTLs and cis-pQTLs

**Paper label:** `fig:ed4` | **Paper file:** `figures/extended_figure_4.png`

| Description | Code       | Data                                              |
| ----------- | ---------- | ------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Mean        | β_rescaled | across variant consequence groups for eQTLs/pQTLs | `chapters/02-analysis/02-variant-effects/02_variant_functional_consequence.ipynb` | `data/intermediate_files/lead_variant_consequence_exploded/` |

---

### Extended Data Figure 5 — Venn diagram of L2G prioritisation reasons

**Paper label:** `fig:ed5` | **Paper file:** `figures/extended_figure_5.png`

| Description                                             | Code                                                                    | Data                                                                |
| ------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------- |
| 4-way Venn: eQTL coloc / pQTL coloc / PAV / nearest TSS | `chapters/02-analysis/03-coloc-l2g/03_using_training_set_for_FDR.ipynb` | `data/intermediate_files/list_of_prioritised_genes_per_CS.parquet/` |

---

### Extended Data Figure 6 — Temporal evolution of L2G confidence and evidence

**Paper label:** `fig:ed6` | **Paper file:** `figures/extended_figure_6.png`

| Description                                              | Code                                                                   | Data                                                                                  |
| -------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Mean max L2G score over time + evidence type proportions | `chapters/02-analysis/03-coloc-l2g/19_temporal_l2g_improvements.ipynb` | `data/intermediate_files/list_of_prioritised_genes_per_CS_with_year_nfe_maf.parquet/` |

---

### Extended Data Figure 7 — Leave-one-out drug target enrichment

**Paper label:** `fig:ed7` | **Paper file:** `figures/extended_figure_7.png`

| Description                                          | Code                                                                          | Data                                                                                            |
| ---------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| OR (95% CI) after leaving out each TA / target class | `chapters/02-analysis/06-target-enrichment/07-split_by_TA_target_class.ipynb` | `data/intermediate_files/chembl_genetic_support_TI_pairs.csv`, `annotated_targets_wide.parquet` |

---

### Extended Data Figure 8 — Target-disease translation success by pleiotropy

**Paper label:** `fig:ed8` | **Paper file:** `figures/extended_figure_8.png`

| Description                                                       | Code                                                                        | Data                                                                  |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Phase transition probabilities by pleiotropy level (Low/Med/High) | `chapters/02-analysis/06-target-enrichment/10-gps-in-clinical-stages.ipynb` | `data/intermediate_files/chembl_evidence_with_rusina_et_al_pharmprj/` |

---

## Supplementary Figures

### Supplementary Methods Figure 1 — Gentropy pipeline schematic

**Paper label:** `fig:sm1` | **Paper file:** `figures/figure_sm1.png`

| Description                                    | Code | Notes                                                        |
| ---------------------------------------------- | ---- | ------------------------------------------------------------ |
| Overview of Gentropy data processing pipelines | —    | Created externally. Not reproducible from code in this repo. |

---

### Supplementary Results Figure 1 — L2G model evaluation

**Paper label:** `fig:sr1` | **Paper file:** `figures/figure_sr1.png`

| Description                                                   | Code                                                                    | Data                                                                         |
| ------------------------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Feature importance, confusion matrix, ROC (AUC=0.95, AP=0.81) | `chapters/02-analysis/03-coloc-l2g/03_using_training_set_for_FDR.ipynb` | `data/25.06/output/l2g_prediction/`, `data/25.06/output/l2g_feature_matrix/` |

---

### Supplementary Results Figure 2 — Colocalisation clusters by MAF bin

**Paper label:** `fig:sr2` | **Paper file:** `figures/figure_sr2.png`

| Description                             | Code                                                                             | Data                                                     |
| --------------------------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------- |
| Cluster count and mean size per MAF bin | `chapters/02-analysis/02-variant-effects/01_lead_variant_effect_filtering.ipynb` | `data/intermediate_files/qualified_lead_variant_effect/` |

---

### Supplementary Results Figure 3 — Beta-effect concordance by MAF

**Paper label:** `fig:sr3` | **Paper file:** `figures/figure_sr3.png`

| Description                                                         | Code                                                                                   | Data                                                       |
| ------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Beta concordance across MAF bins for pleiotropic variants (vPS > 1) | `chapters/02-analysis/02-variant-effects/01_concordanc_of_lead_variant_diseases.ipynb` | `data/intermediate_files/pleiotropy_combined_evidence.csv` |

---

### Supplementary Results Figure 4 — Two-component Gaussian mixture model

**Paper label:** `fig:sr4` | **Paper file:** `figures/figure_sr4.png`

| Description             | Code | Data                               |
| ----------------------- | ---- | ---------------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------- |
| Bimodal distribution of | β    | ² (large vs small effect clusters) | `chapters/02-analysis/04-variant-level-ps/02_clustering_analysis.ipynb` | `data/intermediate_files/cluster_stats_by_year.csv` |

---

## Notes on Data Path Inconsistency

Currently figure scripts look for data in two different places:

- **Figures 1–3:** data stored locally inside
  `chapters/03-manuscript-figures/figure_X/data/`
- **Figures 4–5:** data expected at repo root `data/figure_X/` (Figure 4 also
  reads from `data/intermediate_files/`)

The `data/intermediate_files/` directory holds most of the CSVs needed by figure
scripts but uses different filenames than some scripts expect.

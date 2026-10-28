# Figure-to-code mapping

Every figure in the manuscript, the code that draws it and the tables it reads.
Paths are relative to the repository root; scripts are run from there.

Analysis inputs all live in `data/intermediate_files_refactor/`, written by
`chapters/01-data-preparation` and `chapters/02-analysis-main`. No figure script
fits a model.

---

## Main figures

Run with `tools/run_r.sh <script>`; each writes its PDF next to itself.

| Figure | Script                                                                                                                                           | Reads                                                                                                                                                                           |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1      | `04-figures-main/figure_1/Figure_1_combined.R`, which sources `Figure_1_b_c.R` (b, c), `Figure_1_d.R` (d) and `Figure_1_d_pychart.R` (the donut) | `fig1c_cumulative_discovery.csv`, `qd_sl_eff.csv`, `qm_sl_eff.csv`, `fig1d_gene_index.parquet`, `fig1d_ancestry_donut.csv`, plus `figure_1/assets/` for panel a and the logo    |
| 2      | `04-figures-main/figure_2/figure_2.R`                                                                                                            | `figure_2_a.csv` … `figure_2_d.csv`                                                                                                                                             |
| 3      | `04-figures-main/figure_3/figure_3.R`                                                                                                            | `plot_a.csv`, `plot_b.csv`, `variant_pleiotropy_data_exploded.csv`, `variant_pleiotropy_data_exploded_2.csv`                                                                    |
| 4      | `04-figures-main/figure_4/figure_4.R`                                                                                                            | `Fig4A_stats_gene_pleiotropy.csv`, `Fig4A_stats_variant_pleiotropy.csv`, `Fig4A_stats_gene_coverage.csv`, `gene_pleiotropy_coefficients.csv`, `gene_pleiotropy_by_category.csv` |
| 5      | `04-figures-main/figure_5/figure_5.R`                                                                                                            | `temporal_drug_enrichment_full_chembl.csv`, `drug_enrichment_subsets_vs_full_l2g.csv`, `drug_enrichment_other_resources.csv`, `figure_5b_contrasts.csv`, `figure_5c_curves.csv` |

Panel 1a is a static illustration in `figure_1/assets/Fig1 a (cropped).pdf`, not
generated from data. `figure_2.R` also writes Extended Data Fig. 9. The FinnGen
/ MVP / UKBB reference lines in `Figure_1_b_c.R` are fixed values rather than
read from the data.

The upstream notebooks are `02-analysis-main/01_panoramic.ipynb` (Fig 1),
`02_selective_pressures.ipynb` (Fig 2), `04_variant_pleiotropy.ipynb` (Fig 3),
`05_gene_pleiotropy.ipynb` (Fig 4) and `06_therapeutic_success.ipynb` (Fig 5).

---

## Extended Data figures

In `chapters/05-figures-supplementary/extended_data/`. Notebooks are executed
with `tools/run_chapter.sh`; `ed10` is an R script.

| Figure | Source                                         | Reads                                                                           |
| ------ | ---------------------------------------------- | ------------------------------------------------------------------------------- |
| 1      | — (static asset)                               | `extended_data/assets/extended_figure_1.pdf`, drawn externally                  |
| 2      | `ed02_credible_sets_vs_sample_size.ipynb`      | `lead_variant_effect`, both qualifying credible-set tables                      |
| 3      | `ed03_temporal_measurement_genes.ipynb`        | `ed3_cumulative_discovery.csv`                                                  |
| 4      | `ed04_effect_size_by_consequence.ipynb`        | `variant_consequences`                                                          |
| 5      | `ed05_l2g_venn_diagram.ipynb`                  | `prioritised_genes_per_cs`, both qualifying credible-set tables                 |
| 6      | `ed06_temporal_l2g_confidence.ipynb`           | `prioritised_genes_annotated`                                                   |
| 7      | `ed07_leave_one_out_enrichment.ipynb`          | `prioritised_genes_diseases`, the release evidence, disease and target datasets |
| 8      | `ed08_translation_success_by_pleiotropy.ipynb` | `df_for_enrichment_regression.csv`                                              |
| 9      | `04-figures-main/figure_2/figure_2.R`          | written alongside Figure 2                                                      |
| 10     | `ed10_rare_variant_discovery.R`                | `rare_discovery_over_time.csv`                                                  |

---

## Supplementary figures

In `chapters/05-figures-supplementary/supplementary/`, named by the printed
figure number. Supplementary Figure SR 1 (the L2G model evaluation) and
Supplementary Methods Fig. 1 are external illustrations and are not built here.

| Figure | Source                                    | Reads                                                                            |
| ------ | ----------------------------------------- | -------------------------------------------------------------------------------- |
| SR 2   | `sr02_clusters_by_maf.ipynb`              | `cluster_covariates`                                                             |
| SR 3   | `sr03_concordance_by_maf.ipynb`           | `cluster_covariates`                                                             |
| SR 4   | `sr04_effect_size_mixture.ipynb`          | `lead_variant_effect`, `qualifying_credible_sets`                                |
| SR 5   | `sr05_cluster_disease_vs_ta.ipynb`        | `variant_clusters`                                                               |
| SR 6   | `sr06_success_vs_pleiotropy_counts.ipynb` | `df_for_enrichment_regression.csv`, `ti_pairs_chembl`, `eit_gene_metrics-r1.csv` |

The bootstrap bands of Figure 5c and SR 6 resample rows in a fixed sorted order,
so they can differ slightly from the published bands, which used an unrecorded
Spark row order; the fitted curves are unaffected.

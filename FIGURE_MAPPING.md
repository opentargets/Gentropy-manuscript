# Figure-to-code mapping

Every figure in the manuscript, the code that draws it and the tables it reads.
Paths are relative to the repository root; scripts are run from there.

Analysis inputs all live in `data/intermediate_files_refactor/`, written by
`chapters/01-data-preparation` and `chapters/02-analysis-main`. No figure script
fits a model or reads a legacy file.

Verified against the published PDFs by rendering both at 1,200 px wide and
differencing. `GAPS.md` §3b records the three figures that are not identical and
why.

---

## Main figures

Run with `tools/run_r.sh <script>`; each writes its PDF next to itself.

| Figure | Script                                                                                                                                           | Reads                                                                                                                                                                           | vs. published       |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| 1      | `04-figures-main/figure_1/Figure_1_combined.R`, which sources `Figure_1_b_c.R` (b, c), `Figure_1_d.R` (d) and `Figure_1_d_pychart.R` (the donut) | `fig1c_cumulative_discovery.csv`, `qd_sl_eff.csv`, `qm_sl_eff.csv`, `fig1d_gene_index.parquet`, `fig1d_ancestry_donut.csv`, plus `figure_1/assets/` for panel a and the logo    | identical           |
| 2      | `04-figures-main/figure_2/figure_2.R`                                                                                                            | `figure_2_a.csv` … `figure_2_d.csv`                                                                                                                                             | identical           |
| 3      | `04-figures-main/figure_3/figure_3.R`                                                                                                            | `plot_a.csv`, `plot_b.csv`, `variant_pleiotropy_data_exploded.csv`, `variant_pleiotropy_data_exploded_2.csv`                                                                    | 0.03 %              |
| 4      | `04-figures-main/figure_4/figure_4.R`                                                                                                            | `Fig4A_stats_gene_pleiotropy.csv`, `Fig4A_stats_variant_pleiotropy.csv`, `Fig4A_stats_gene_coverage.csv`, `gene_pleiotropy_coefficients.csv`, `gene_pleiotropy_by_category.csv` | 0.4 %               |
| 5      | `04-figures-main/figure_5/figure_5.R`                                                                                                            | `temporal_drug_enrichment_full_chembl.csv`, `drug_enrichment_subsets_vs_full_l2g.csv`, `drug_enrichment_other_resources.csv`, `figure_5b_contrasts.csv`, `figure_5c_curves.csv` | 3.3 %, panel c only |

Panel 1a is a static illustration in `figure_1/assets/Fig1 a (cropped).pdf`, not
generated from data. `figure_2.R` also writes Extended Data Fig. 9.

The FinnGen / MVP / UKBB reference lines in `Figure_1_b_c.R` are fixed values
rather than read from the data — deliberate, and the only such case among the
main figures.

The upstream notebooks are `02-analysis-main/01_panoramic.ipynb` (Fig 1),
`02_selective_pressures.ipynb` (Fig 2), `04_variant_pleiotropy.ipynb` (Fig 3),
`05_gene_pleiotropy.ipynb` (Fig 4) and `06_therapeutic_success.ipynb` (Fig 5).

---

## Extended Data figures

In `chapters/05-figures-supplementary/extended_data/`. Notebooks are executed
with `tools/run_chapter.sh`; `ed10` is an R script.

| Figure | Source                                         | Reads                                                                           | vs. published |
| ------ | ---------------------------------------------- | ------------------------------------------------------------------------------- | ------------- |
| 1      | — (static asset)                               | `extended_data/assets/extended_figure_1.pdf`, drawn externally                  | n/a           |
| 2      | `ed02_credible_sets_vs_sample_size.ipynb`      | `lead_variant_effect`, both qualifying credible-set tables                      | identical     |
| 3      | `ed03_temporal_measurement_genes.ipynb`        | `ed3_cumulative_discovery.csv`                                                  | identical     |
| 4      | `ed04_effect_size_by_consequence.ipynb`        | `variant_consequences`                                                          | identical     |
| 5      | `ed05_l2g_venn_diagram.ipynb`                  | `prioritised_genes_per_cs`, both qualifying credible-set tables                 | identical     |
| 6      | `ed06_temporal_l2g_confidence.ipynb`           | `prioritised_genes_annotated`                                                   | identical     |
| 7      | `ed07_leave_one_out_enrichment.ipynb`          | `prioritised_genes_diseases`, the release evidence, disease and target datasets | identical     |
| 8      | `ed08_translation_success_by_pleiotropy.ipynb` | `df_for_enrichment_regression.csv`                                              | identical     |
| 9      | `04-figures-main/figure_2/figure_2.R`          | the panel demoted from Figure 2                                                 | identical     |
| 10     | `ed10_rare_variant_discovery.R`                | `rare_discovery_over_time.csv`                                                  | identical     |

---

## Supplementary figures

Of Supplementary Results Figures 1–6, SR 2–6 have been carried into the
refactored pipeline. Supplementary Methods Fig. 1 is an external illustration.
See `GAPS.md`.

Numbers below are the **printed** figure numbers, which is how the figures are
cited in the text. **SR 5 and SR 6 have their asset filenames and label names
swapped**, so watch the last column:

| Printed as | Source                                                  | Manuscript asset     | Label     |
| ---------- | ------------------------------------------------------- | -------------------- | --------- |
| SR 1       | — (no source)                                           | `figure_sr1.png`     | `fig:sr1` |
| SR 2       | `supplementary/sr02_clusters_by_maf.ipynb`              | `figure_sr2.png`     | `fig:sr2` |
| SR 3       | `supplementary/sr03_concordance_by_maf.ipynb`           | `figure_sr3.png`     | `fig:sr3` |
| SR 4       | `supplementary/sr04_effect_size_mixture.ipynb`          | `figure_sr4.png`     | `fig:sr4` |
| SR 5       | `supplementary/sr05_cluster_disease_vs_ta.ipynb`        | **`figure_sr6.pdf`** | `fig:sr6` |
| SR 6       | `supplementary/sr06_success_vs_pleiotropy_counts.ipynb` | **`figure_sr5.pdf`** | `fig:sr5` |

The cluster scatter appears at `sections/supplementary_results.tex:283` and the
ten-panel figure at line 1116, so LaTeX numbers the cluster scatter 5 and the
ten-panel 6 — the opposite of what their filenames and labels say. The rendered
prose is correct throughout, because every citation goes through `\ref`. Only
the asset names mislead. **Notebooks and outputs in this repository are named
after the printed number**, so `supplementary/figure_sr5.pdf` here is the
cluster scatter and corresponds to the manuscript's `figures/figure_sr6.pdf`.
Copying a PDF across without swapping the name would silently put the wrong
figure in the paper.

| Printed as | Inputs                                                                           | Match                                       |
| ---------- | -------------------------------------------------------------------------------- | ------------------------------------------- |
| SR 1       | training set unavailable                                                         | not built                                   |
| SR 2       | `cluster_covariates`                                                             | every bar matches                           |
| SR 3       | `cluster_covariates`                                                             | 5 of 7 bins match; see the chapter README   |
| SR 4       | `lead_variant_effect`, `qualifying_credible_sets`                                | same points, split and curve heights        |
| SR 5       | `variant_clusters`                                                               | new; no published PDF to match              |
| SR 6       | `df_for_enrichment_regression.csv`, `ti_pairs_chembl`, `eit_gene_metrics-r1.csv` | 0.94% of pixels, all in the bootstrap bands |

SR 5 was added for the round-1 response (Reviewer 1, minor comment 9) and is
rebuilt here on the Supplementary Table 9 therapeutic-area order, so its
coordinate count differs from the round-1 draft: 226 rather than 219. Design
unchanged — two panels, linear and log, marker area ∝ √n, size key.

SR 4's published figure was drawn from a hand-pasted vector of 20 effect sizes
with no variant id attached; it is `9_22124745_C_G`, recovered by matching the
vector against the release. It also needs random EM initialisation — the k-means
default behind the Supplementary Results 6 counts finds a worse local optimum on
this variant.

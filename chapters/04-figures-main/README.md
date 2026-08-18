# Main text figures

Run from the repository root, after `chapters/02-analysis-main` has written its
inputs:

```
tools/run_r.sh chapters/04-figures-main/figure_2/figure_2.R
```

Every script reads `data/intermediate_files_refactor` and writes its PDF next to
itself. `figure_2.R` also writes Extended Data Fig. 9, which is the panel
demoted from Figure 2.

| Figure | Script                                                                                            | Inputs                                                                                                                    |
| ------ | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 1      | `figure_1/Figure_1_combined.R` (sources `Figure_1_b_c.R`, `Figure_1_d.R`, `Figure_1_d_pychart.R`) | `qd_sl_eff.csv`, `qm_sl_eff.csv`, `fig1c_cumulative_discovery.csv`, the pleiotropy map parquet                            |
| 2      | `figure_2/figure_2.R`                                                                             | `figure_2_a.csv` … `figure_2_d.csv`                                                                                       |
| 3      | `figure_3/figure_3.R`                                                                             | `plot_a.csv`, `plot_b.csv`, `variant_pleiotropy_data_exploded.csv`, `..._2.csv`                                           |
| 4      | `figure_4/figure_4.R`                                                                             | `Fig4A_stats_*.csv`, `gene_pleiotropy_full_model.csv`, `gene_pleiotropy_by_category.csv`                                  |
| 5      | `figure_5/figure_5.R`                                                                             | `temporal_drug_enrichment_full_chembl.csv`, `drug_enrichment_subsets_vs_full_l2g.csv`, `df_for_enrichment_regression.csv` |

Panel 1a is an illustration, not generated here; `figure_1/assets/` holds it.

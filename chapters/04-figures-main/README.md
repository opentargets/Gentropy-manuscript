# Main text figures

Run from the repository root, after `chapters/02-analysis-main` has written its
inputs:

```
tools/run_r.sh chapters/04-figures-main/figure_2/figure_2.R
```

Every script reads `data/intermediate_files_refactor` and writes its PDF next to
itself. `figure_2.R` also writes Extended Data Fig. 9, which is the panel
demoted from Figure 2.

| Figure | Script                                                                                            | Inputs                                                                                                                                                                          |
| ------ | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1      | `figure_1/Figure_1_combined.R` (sources `Figure_1_b_c.R`, `Figure_1_d.R`, `Figure_1_d_pychart.R`) | `fig1c_cumulative_discovery.csv`, `qd_sl_eff.csv`, `qm_sl_eff.csv`, `fig1d_gene_index.parquet`, `fig1d_ancestry_donut.csv`                                                      |
| 2      | `figure_2/figure_2.R`                                                                             | `figure_2_a.csv` … `figure_2_d.csv`                                                                                                                                             |
| 3      | `figure_3/figure_3.R`                                                                             | `plot_a.csv`, `plot_b.csv`, `variant_pleiotropy_data_exploded.csv`, `..._2.csv`                                                                                                 |
| 4      | `figure_4/figure_4.R`                                                                             | `Fig4A_stats_*.csv`, `gene_pleiotropy_coefficients.csv`, `gene_pleiotropy_by_category.csv`                                                                                      |
| 5      | `figure_5/figure_5.R`                                                                             | `temporal_drug_enrichment_full_chembl.csv`, `drug_enrichment_subsets_vs_full_l2g.csv`, `drug_enrichment_other_resources.csv`, `figure_5b_contrasts.csv`, `figure_5c_curves.csv` |

## Figure 3 was rebuilt under the lead_vPS redefinition

Panels a and b follow the new cluster representative; panel c switched from
`rescaledStatistics.minorAlleleEstimatedBeta` to the harmonised effect-allele
beta (`directionOfEffect * absEstimatedBeta`) and now draws only the
contributing credible sets — those of studies mapped to exactly one disease
term, which are what lead_vPS and the concordance are computed over. Against the
published `figures/figure_3.pdf` the rebuild differs in 1.01% of pixels at 3x
rasterisation, where the unchanged pipeline differed in 0.03%; panels a and b
account for nearly all of it. Reasons and before/after numbers:
`chapters/02-analysis-main/README.md`.

The `chi2Stat` rank-key experiment of 2026-08-22 was reverted the same day; the
figure was rebuilt under it (**0 of 1,365,525 pixels differ**) and `plot_a.csv`
/ `plot_b.csv` are back to byte-identical after the revert, so no rebuild
stands. Note that `~/Projects/manuscript_gentropy/figures/figure_3.pdf` was
replaced with the rebuilt figure on 2026-08-22, so it is no longer the original
published asset and a comparison against it now returns 0.00% by construction —
the 1.01% figure quoted below was measured against the original.

The sign-gate amendment to lead_vPS does **not** touch this figure either.
`plot_a.csv`, `plot_b.csv` and `cluster_covariates` are value-identical across
the amendment; `04_variant_pleiotropy.ipynb` asserts R4.10-R4.13 and all twelve
panel-b coefficients against literals, so a drift would fail the notebook rather
than pass unnoticed. Panel c is unaffected too, because the APOE export already
drops credible sets with a null `originalBeta`, which is exactly what the gate
removes.

Panel 1a is an illustration, not generated from data; `figure_1/assets/` holds
it as a static PDF and `Figure_1_combined.R` rasterises it into the top strip.
Everything else in Figure 1 is built from `data/intermediate_files_refactor`.

`Figure_1_combined.R` draws through `quartz()` on macOS and falls back to
`cairo_pdf()` elsewhere; the panel-a strip is rasterised with `sips` on macOS
and with `pdftools::pdf_convert()` elsewhere.

## These scripts only draw

No model is fitted here. Figure 4b used to re-fit the nine univariate and one
joint negative-binomial models that `02-analysis-main/05_gene_pleiotropy.ipynb`
already writes to `gene_pleiotropy_coefficients.csv`, and Figure 5c used to run
400 logistic fits and lowess smooths that now live in
`06_therapeutic_success.ipynb` as `figure_5c_curves.csv`. Keeping a published
number in one place means the figure cannot drift from the reported value, and
the whole chapter now rebuilds in about 20 seconds.

Two reference values are read rather than pasted in for the same reason: the
Figure 5a "2025 enrichment" line is the `full_l2g` odds ratio, and the Figure 5b
significance stars come from the `fdr` column of `figure_5b_contrasts.csv`.

The exception, left deliberately: the FinnGen / MVP / UKBB reference lines in
`Figure_1_b_c.R` are fixed values, not read from the data.

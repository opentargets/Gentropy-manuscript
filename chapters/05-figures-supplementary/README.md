# Supplementary figures

## Extended Data figures

Run from the repository root, after `chapters/02-analysis-main`. The notebooks
go in one pass; ED 10 is an R script and ED 9 falls out of Figure 2.

```
tools/run_chapter.sh chapters/05-figures-supplementary/extended_data
tools/run_r.sh chapters/05-figures-supplementary/extended_data/ed10_rare_variant_discovery.R
```

ED 7 is the slow one, about fifteen minutes; the rest are a few minutes each.
Every figure here was checked against the published PDF by rendering both at
1,200 px wide, and ED 2-10 are pixel-identical.

| Figure | Source                                         | Inputs                                                                                  |
| ------ | ---------------------------------------------- | --------------------------------------------------------------------------------------- |
| ED 1   | — (static asset)                               | `extended_data/assets/extended_figure_1.pdf`, drawn externally; see GAPS.md             |
| ED 2   | `ed02_credible_sets_vs_sample_size.ipynb`      | `lead_variant_effect`, both qualifying credible-set tables                              |
| ED 3   | `ed03_temporal_measurement_genes.ipynb`        | `ed3_cumulative_discovery.csv`                                                          |
| ED 4   | `ed04_effect_size_by_consequence.ipynb`        | `variant_consequences`                                                                  |
| ED 5   | `ed05_l2g_venn_diagram.ipynb`                  | `prioritised_genes_per_cs`                                                              |
| ED 6   | `ed06_temporal_l2g_confidence.ipynb`           | `prioritised_genes_annotated`                                                           |
| ED 7   | `ed07_leave_one_out_enrichment.ipynb`          | `prioritised_genes_diseases`                                                            |
| ED 8   | `ed08_translation_success_by_pleiotropy.ipynb` | `df_for_enrichment_regression.csv`; also writes the two `ed8_phase_transition_*` tables |
| ED 9   | `chapters/04-figures-main/figure_2/figure_2.R` | the panel demoted from Figure 2                                                         |
| ED 10  | `ed10_rare_variant_discovery.R`                | `rare_discovery_over_time.csv`                                                          |

## Supplementary Results figures

`supplementary/` holds SR Figs 2–6. SR 1 (L2G model evaluation) is a Weights &
Biases screenshot and needs the training set and the held-out split, which are
not available; see GAPS.md. Run after `chapters/02-analysis-main`.

```
tools/run_chapter.sh chapters/05-figures-supplementary/supplementary
```

SR 6 is the slow one, about three minutes of bootstrapping; the rest are
seconds.

| Figure | Source                                    | Inputs                                                                           |
| ------ | ----------------------------------------- | -------------------------------------------------------------------------------- |
| SR 1   | — (no source)                             | Weights & Biases screenshot; training set unavailable                            |
| SR 2   | `sr02_clusters_by_maf.ipynb`              | `cluster_covariates`                                                             |
| SR 3   | `sr03_concordance_by_maf.ipynb`           | `cluster_covariates`                                                             |
| SR 4   | `sr04_effect_size_mixture.ipynb`          | `lead_variant_effect`, `qualifying_credible_sets`                                |
| SR 5   | `sr05_cluster_disease_vs_ta.ipynb`        | `variant_clusters`                                                               |
| SR 6   | `sr06_success_vs_pleiotropy_counts.ipynb` | `df_for_enrichment_regression.csv`, `ti_pairs_chembl`, `eit_gene_metrics-r1.csv` |

**SR 5 and SR 6 are numbered by what the document prints, not by the
manuscript's filenames, which are swapped for these two.** The cluster scatter
appears first in `supplementary_results.tex` (line 283) so it prints as
Supplementary Figure 5, but its asset is `figures/figure_sr6.pdf` and its label
`fig:sr6`; the ten-panel figure prints as 6 from `figure_sr5.pdf` and `fig:sr5`.
The rendered prose is right — every citation goes through `\ref` — but do not
copy a PDF from here into `figures/` without swapping the name.
`FIGURE_MAPPING.md` carries the full mapping.

`cluster_covariates` is written by
`chapters/02-analysis-main/04_variant_pleiotropy` — the same per-cluster table
Figure 3 is drawn from, one row per cluster at its representative lead variant.

**SR 2 and SR 3 were rebuilt on 2026-08-22**, after the lead_vPS redefinition
and its sign-gate amendment. `chapters/03-analysis-supplementary/README.md`
carries the before/after values; both notebooks close with a note on what
changed and why.

- **SR 2** moved with the cluster representative alone — it counts clusters at
  the representative's MAF, so no concordance definition reaches it. Bars 496 /
  2,372 / 1,868 / 3,529 / 3,843 / 3,867 / 4,065 became **496 / 2,360 / 1,849 /
  3,579 / 3,823 / 3,877 / 4,055**; 20,039 of 20,041 fall inside the half-open
  bins, as before.
- **SR 3** was moved onto the current protocol at the author's direction:
  `signedLeadDirectionalConcordance` over the `signedLeadVPS > 1` universe. It
  no longer plots the published quantity, so no pixel comparison against
  `figures/figure_sr3.png` is meaningful. Universe 5,188 clusters -> **2,166**,
  with **3,019** representatives excluded because nothing contributes.

Both were rebuilt under the `chi2Stat` rank-key experiment of 2026-08-22 and
came back **pixel-identical** (0 of 2,719,536 and 0 of 2,717,398). That
experiment was reverted the same day and `cluster_covariates` is byte-identical
again, so the values above stand and neither PDF needs another rebuild.

SR 4 and SR 5 were not affected: SR 4 reads `lead_variant_effect` and
`qualifying_credible_sets`, SR 5 reads `variant_clusters`, and none of the three
moved. Verified against their stored outputs rather than assumed.

**`figure_sr6.pdf` is missing from `supplementary/`.** The `sr06` notebook is
fully executed, has no error cells, and its own output reads
`saved figure_sr6.pdf`, so it built at some point and the file has since been
removed. Re-run `sr06_success_vs_pleiotropy_counts.ipynb` before the assets are
collected — it needs `eit_gene_metrics-r1.csv` for panels 3-10, as noted below.

Each notebook closes with a comparison against the published figure. In summary:

- **SR 2** (clusters per MAF bin) reproduces bin for bin.
- **SR 3** (directional concordance by MAF bin) reproduces in the five lower-MAF
  bins; the two highest sit below the published points, for the same unresolved
  reason as S6.07 and S6.08 — see
  `chapters/03-analysis-supplementary/README.md`.
- **SR 4** (two-component mixture at one variant) reproduces, once the variant
  is identified — the published figure was drawn from a hand-pasted vector with
  no id attached, matched back to `9_22124745_C_G`. It needs random EM
  initialisation: the k-means default behind the S6 counts lands on a worse
  local optimum here.
- **SR 5** (diseases against therapeutic areas per cluster) is new for the
  round-1 response, so there is no published PDF to match; it is rebuilt on the
  Supplementary Table 9 therapeutic-area order.
- **SR 6** (ten trait counts against clinical success) reproduces; 0.94% of
  pixels differ, all in the bootstrap bands and the LOWESS curves, because the
  resampling follows row order and the published table's order is unrecoverable
  — the same caveat GAPS.md records for Figure 5c.

**SR 6 has one input this pipeline cannot rebuild**: `eit_gene_metrics-r1.csv`,
the per-gene effective-independent-trait counts, comes from the Supplementary
Results 14.3/14.4 analysis still sitting in
`chapters/_legacy/06-review-r1/effective-independent-traits/`. Panels 1–2 need
nothing outside the refactored pipeline; panels 3–10 need that file.

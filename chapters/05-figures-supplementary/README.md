# Supplementary figures

## Extended Data figures

Run from the repository root, after `chapters/02-analysis-main`:

```
uv run jupyter nbconvert --to notebook --execute --inplace \
  chapters/05-figures-supplementary/extended_data/ed02_credible_sets_vs_sample_size.ipynb
tools/run_r.sh chapters/05-figures-supplementary/extended_data/ed10_rare_variant_discovery.R
```

| Figure | Source                                         | Inputs                                                                                           |
| ------ | ---------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| ED 1   | —                                              | flowchart, drawn externally                                                                      |
| ED 2   | `ed02_credible_sets_vs_sample_size.ipynb`      | `lead_variant_effect`, both qualifying credible-set tables                                       |
| ED 3   | `ed03_temporal_measurement_genes.ipynb`        | `prioritised_genes_measurements`                                                                 |
| ED 4   | `ed04_effect_size_by_consequence.ipynb`        | `variant_consequences`                                                                           |
| ED 5   | `ed05_l2g_venn_diagram.ipynb`                  | `prioritised_genes_per_cs`                                                                       |
| ED 6   | `ed06_temporal_l2g_confidence.ipynb`           | `prioritised_genes_annotated`                                                                    |
| ED 7   | `ed07_leave_one_out_enrichment.ipynb`          | `prioritised_genes_diseases`                                                                     |
| ED 8   | `ed08_translation_success_by_pleiotropy.ipynb` | `df_for_enrichment_regression.csv` and the phase-transition tables from Supplementary Results 10 |
| ED 9   | `chapters/04-figures-main/figure_2/figure_2.R` | the panel demoted from Figure 2                                                                  |
| ED 10  | `ed10_rare_variant_discovery.R`                | `rare_discovery_over_time.csv`                                                                   |

## Supplementary Results figures

`supplementary/` holds SR Figs 1–6. SR 1 (L2G model evaluation) needs the
training set and the held-out split, which are not available; see GAPS.md.

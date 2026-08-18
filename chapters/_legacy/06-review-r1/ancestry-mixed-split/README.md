# R1 — `ancestry-mixed-split`

Reviewer 1: the ancestry split was binary (`EUR` = predominant ancestry ≥90%
non-Finnish European, `non-EUR` = everything else), so an 89%-European
pan-ancestry meta-analysis and FinnGen both counted as "non-EUR". The claim that
~30% of disease-associated genes were first found in non-EUR studies is
therefore compatible with the growth being driven entirely by larger European
meta-analyses.

## Reclassification

| Label     | Rule                                                                    |
| --------- | ----------------------------------------------------------------------- |
| `EUR`     | one ancestry reaches ≥90% of the sample and it is non-Finnish European  |
| `non-EUR` | one ancestry reaches ≥90% and it is not NFE — Finnish counts as non-EUR |
| `mixed`   | no single ancestry reaches 90% — pan-ancestry meta-analyses land here   |
| `rare`    | MAF < 0.01, reported separately, orthogonal to ancestry (unchanged)     |

Studies whose ancestry cannot be determined are called `EUR`: 31 of 100,526 GWAS
(0.03%) — 17 with a null or empty `ldPopulationStructure`, 13 listing a single
`nfe` population with a null `relativeSampleSize`, and 1 listing four
populations with all sizes null. Defaulting them to `EUR` is conservative (it
can only shrink the non-EUR and mixed contributions) and at 0.03% cannot move
any reported number. The `ldStructureNote` column flags which studies were
defaulted.

Nesting order for the stacked tiers: **1) EUR common, 2) non-EUR common, 3)
mixed common, 4) rare** (any ancestry). A layer's height is what that tier
reaches and the tiers below it do not.

## Notebook

`01_ancestry_reclassification.ipynb` — reclassifies every GWAS study in the
25.06 study index, rebuilds the row-level gene-discovery table with the new
labels, and recomputes per category and per year (2006–2024): unique
disease-associated genes first discovered, total unique gene–disease
associations, the first-discovery curve underlying Figure 1c, and the same for
measurements (Extended Data Figure 3).

Three views are produced for each metric:

- **nested tiers** — what the stacked bars encode; tier _k_ uses only the first
  _k_ ancestry labels (common variants), the last tier adds rare variants from
  any ancestry.
- **marginal** — discoveries within a single ancestry × frequency stratum,
  independent of the others.
- **attribution** — per stratum: entities reachable from it, exclusive to it,
  and first discovered in it (ties broken in nesting order, so EUR wins). Also
  computed under the old binary rule for a direct old-vs-new comparison.

Cumulative plots (stacked bars + tier lines) are drawn inline in the notebook
for inspection. They are drafts, not manuscript figures, and are not saved to
disk.

## Inputs (all local, from `data/`)

- `data/25.06/output/study`, `data/25.06/output/credible_set`
- `data/intermediate_files/list_of_prioritised_genes_per_CS.parquet`
- `data/intermediate_files/lead_variant_effect`
- `data/intermediate_files/qualifying_credible_sets`,
  `qualifying_measurement_credible_sets`

## Outputs

Written to `data/intermediate_files/`, all suffixed `-r1` so nothing overwrites
the original analysis:

| File                                                             | Use                                                         |
| ---------------------------------------------------------------- | ----------------------------------------------------------- |
| `study_ancestry_classification-r1.csv`                           | per-study ancestry label lookup, reusable by other R1 work  |
| `study_ancestry_counts-r1.csv`                                   | study counts per class, per study subset                    |
| `ancestry_composition-r1.csv`                                    | what the `non-EUR` and `mixed` classes are made of          |
| `list_of_prioritised_genes_per_CS_with_year_ancestry-r1.parquet` | row-level table, superset of the original `*_nfe_maf` one   |
| `l2g_diseases_full-r1.csv`                                       | Figure 1c replot input                                      |
| `l2g_measurements_full-r1.csv`                                   | Extended Data Figure 3 replot input                         |
| `fig1c_cumulative_discovery_nested-r1.csv`                       | Fig 1c tiers and layer heights, per year                    |
| `fig1c_cumulative_discovery_marginal-r1.csv`                     | Fig 1c per-stratum curves, per year                         |
| `fig1c_discovery_attribution-r1.csv`                             | reachable / exclusive / first-discovery counts, old vs new  |
| `ed3_cumulative_discovery_nested-r1.csv`                         | ED Fig 3 tiers and layer heights, per year                  |
| `ed3_cumulative_discovery_marginal-r1.csv`                       | ED Fig 3 per-stratum curves, per year                       |
| `ed3_discovery_attribution-r1.csv`                               | ED Fig 3 reachable / exclusive / first-discovery counts     |
| `ancestry_discovery_headline-r1.csv`                             | final-year tier totals and increments for the response text |

Figure scripts in `chapters/03-manuscript-figures/` are deliberately left
untouched; replotting is a separate step.

The `l2g_diseases_full-r1.csv` column layout is a superset of the original
`chapters/03-manuscript-figures/figure_1/l2g_diseases_full.csv`, so the existing
`Figure_1_b_c.R` reads it unchanged (old flags `nfe_common`, `non_nfe_common`,
`rare` are preserved); the new flags `eur_common`, `noneur_common`,
`mixed_common` and the `ancestryClass` / `freqClass` columns are added
alongside.

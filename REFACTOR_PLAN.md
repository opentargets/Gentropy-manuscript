# Refactor plan

Goal: one simple, runnable path from downloaded data to every number, figure and
supplementary table in the manuscript, with nothing computed twice and nothing
computed in a place the manuscript does not use.

Manuscript (read-only, never modified by this repo):
`~/Projects/manuscript_gentropy/`.

## Layout

| Directory                             | Contents                                                                 |
| ------------------------------------- | ------------------------------------------------------------------------ |
| `chapters/00-data-download/`          | Data acquisition. `01_download_data_to_local_repo.ipynb` is unchanged.   |
| `chapters/01-data-preparation/`       | One notebook per canonical dataset. Nothing else writes these.           |
| `chapters/02-analysis-main/`          | One notebook per Results subsection.                                     |
| `chapters/03-analysis-supplementary/` | One notebook per Supplementary Results section.                          |
| `chapters/04-figures-main/`           | Figures 1–5.                                                             |
| `chapters/05-figures-supplementary/`  | Extended Data Figs 2–10, Supplementary Results Figs 1–6.                 |
| `chapters/06-supplementary-tables/`   | Supplementary Tables 1–16.                                               |
| `chapters/r-env/`                     | Shared R library (`renv`). Use `tools/run_r.sh`.                         |
| `chapters/_legacy/`                   | The pre-refactor chapters, kept for reference. Not part of the pipeline. |
| `src/manuscript_methods/`             | Existing method library. Pre-existing modules are not modified.          |
| `tools/`                              | `expected_numbers.tsv`, `check_numbers.py`, `run_r.sh`.                  |
| `results/`                            | One JSON per analysis notebook: the numbers it computed.                 |

## Data

- `data/25.06/`, `data/intermediate_files/` — **read-only during the refactor.**
  The existing intermediates are the baseline the refactor is checked against.
- `data/intermediate_files_refactor/` — everything the refactored pipeline
  writes.

Canonical dataset names, each written by exactly one notebook in
`01-data-preparation`:

| Dataset                                                                                                                    | Producer                       |
| -------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| `lead_variant_effect`                                                                                                      | `01_lead_variant_effect`       |
| `replicated_gwas_cs`, `replicated_molqtl_cs`                                                                               | `02_replicated_credible_sets`  |
| `efo_therapeutic_area`, `study_therapeutic_areas`                                                                          | `03_therapeutic_areas`         |
| `qualifying_{gwas,measurement}_studies`, `qualifying_{,measurement_}credible_sets`                                         | `04_qualifying_studies_and_cs` |
| `prioritised_genes_per_cs`, `study_annotation`, `prioritised_genes_annotated`, `prioritised_genes_{diseases,measurements}` | `05_l2g_prioritised_genes`     |
| `gene_table`                                                                                                               | `06_gene_level_table`          |
| `variant_clusters`, `cluster_membership`                                                                                   | `07_variant_clusters`          |
| `replicated_lead_variants`, `replicated_lead_variants_common`                                                              | `08_replicated_lead_variants`  |
| `variant_consequences`                                                                                                     | `09_variant_consequences`      |
| `gene_sets`                                                                                                                | `10_gene_sets`                 |
| `ti_pairs_chembl`, `ti_pairs_pharmaprojects`, `l2g_indirect_assoc_{all,pav}`                                               | `11_drug_pair_tables`          |
| `rg_matrix`                                                                                                                | `12_genetic_correlation`       |

`gene_table` replaces `genes_therapeutic_areas` and `genes_pleiotropy`, which in
the pre-refactor repo were only produced in `playground/` from another machine's
local paths.

`variant_clusters` uses the ontology-resolved `diseaseIds` column, matching the
numbers in the current manuscript text (20,041 clusters; 6,617 multi-disease;
mean 2.14).

## Conventions

- Notebooks read data via paths relative to the repository root and are executed
  from the repository root (`uv run jupyter nbconvert --execute`).
- Only one therapeutic-area hierarchy is used, defined in one place.
- No `-r1` suffixes. Where a `-r1` variant was the current truth, it becomes the
  only version.
- Every analysis notebook's last cell writes `results/<notebook-id>.json`.
- Figures are written next to their script and are not copied into the
  manuscript repo.

## Verification

`tools/expected_numbers.tsv` holds every number claimed in the manuscript
Results plus the headline numbers of each Supplementary Results section, with
the `.tex` file it came from. `tools/check_numbers.py` compares it against
`results/*.json` and writes `REPRODUCIBILITY.md`.

A notebook is done when its numbers pass, or when the mismatch is written up in
`REPRODUCIBILITY.md`.

## Not covered yet

`GAPS.md` lists the inputs that are missing and the parts of the manuscript that
have no code in this repository. Those are flagged, not guessed at.

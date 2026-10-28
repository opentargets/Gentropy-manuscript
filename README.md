# Gentropy Manuscript Analysis

Code for **The Human Pleiotropic Map of GWAS Associations and Therapeutic
Implications**
([preprint](https://www.biorxiv.org/content/10.64898/2026.04.28.721048v1)).

## Requirements

Linux or macOS, 40 GB RAM, 40 GB disk. Python ≥3.11 <3.14 with
[uv](https://docs.astral.sh/uv/), Java 11, R ≥4.3, the gcloud SDK.

```bash
make dev                                  # python environment, ~10 min
gcloud auth application-default login     # to download the release
```

R packages live in `chapters/r-env`; run R through `tools/run_r.sh`, which
points at them. Restore the library with
`R_LIBS_SITE= Rscript -e 'renv::restore(project="chapters/r-env")'` — the figure
scripts need it, and so does
`03-analysis-supplementary/08_enrichment_bias.ipynb`, which fits its
mixed-effects models with `lme4::glmer` through the same wrapper.

## Running it

Each chapter is run in order, from the repository root.

```bash
tools/run_chapter.sh chapters/01-data-preparation
tools/run_chapter.sh chapters/02-analysis-main
tools/run_r.sh chapters/04-figures-main/figure_5/figure_5.R
uv run python tools/check_numbers.py
```

| Chapter                     | What it does                                                                                                               |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `00-data-download`          | Downloads the Open Targets 25.06 release and the project inputs from Zenodo (see its README).                              |
| `01-data-preparation`       | Thirteen notebooks, one canonical dataset each, written to `data/intermediate_files_refactor/`. Nothing else writes these. |
| `02-analysis-main`          | Six notebooks, one per Results subsection. Each writes its numbers to `results/`.                                          |
| `03-analysis-supplementary` | One notebook per Supplementary Results section.                                                                            |
| `04-figures-main`           | Figures 1–5.                                                                                                               |
| `05-figures-supplementary`  | Extended Data Figures 2–10 (`extended_data/`) and Supplementary Figures SR 2–6 (`supplementary/`).                         |
| `06-supplementary-tables`   | The supplementary table sheets.                                                                                            |
| `r-env`                     | Shared R library.                                                                                                          |

To skip chapter 01, fetch `intermediates.tar.gz` from the Zenodo deposit as well
(see `chapters/00-data-download`) and start at chapter 02.

`src/manuscript_methods/` holds the shared code: the lead-variant-effect
pipeline, the therapeutic-area hierarchy and paths (`paper.py`), ancestry and
discovery curves (`discovery.py`), colocalisation clustering (`clusters.py`) and
drug-target enrichment statistics (`enrichment.py`).

## Where each part of the manuscript is computed

Figures are mapped to their scripts and inputs in `FIGURE_MAPPING.md`.

| Manuscript                                                           | Notebook                                                                       |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Results 1 — panoramic view across complex-trait GWAS                 | `02-analysis-main/01_panoramic`                                                |
| Results 2 — selective pressures, concordant effects, directionality  | `02-analysis-main/02_selective_pressures`                                      |
| Results 3 — colocalisation and effector-gene prediction              | `02-analysis-main/03_colocalisation_l2g`                                       |
| Results 4 — variant-level pleiotropy                                 | `02-analysis-main/04_variant_pleiotropy`                                       |
| Results 5 — gene-level pleiotropy                                    | `02-analysis-main/05_gene_pleiotropy`                                          |
| Results 6 — genetic evidence and therapeutic success                 | `02-analysis-main/06_therapeutic_success`                                      |
| Supplementary Results 1–10                                           | `03-analysis-supplementary/01_…` to `10_…`, numbered as the sections           |
| Supplementary Results 11.1–11.3 and 11.4–11.5                        | `03-analysis-supplementary/11_criterion_validation`, `12_external_replication` |
| Supplementary Results 12                                             | `03-analysis-supplementary/13_ancestry_and_sample_size`                        |
| Supplementary Results 14.1–14.2                                      | `03-analysis-supplementary/14_genetic_correlation`                             |
| Immunity against infection, directionality (response to reviewers)   | `03-analysis-supplementary/15_immunity_infection_directionality`               |
| ChEMBL against Pharmaprojects approved pairs (response to reviewers) | `03-analysis-supplementary/16_chembl_vs_pharmaprojects`                        |
| Supplementary tables                                                 | `06-supplementary-tables/` (below)                                             |

Supplementary table sheets keep this repository's own file numbers, which
predate the printed numbering:

| Printed                     | Built by                  | File in `06-supplementary-tables/`           |
| --------------------------- | ------------------------- | -------------------------------------------- |
| ST1 studies                 | `01_supplementary_tables` | `sheets/ST1_studies.csv`                     |
| ST2 TA distribution         | `01_supplementary_tables` | `sheets/ST16_ta_distribution.csv`            |
| ST3 effector gene list      | `05_effector_gene_list`   | `sheets/ST18_effector_gene_list.csv`         |
| ST4 L2G training set        | `06_l2g_training_set`     | `sheets/ST19_l2g_training_set.csv`           |
| ST5 cluster membership      | `01_supplementary_tables` | `sheets/ST15_cluster_membership.csv`         |
| ST6 discordant variants     | `01_supplementary_tables` | `sheets/ST2_discordant_variants.csv`         |
| ST7 gene-disease with gPS   | `01_supplementary_tables` | `sheets/ST14_gene_disease_with_gps.csv`      |
| ST8 GSEA                    | compiled by hand          | `assets/ST8_-_GSEA_results.xlsx`             |
| ST9 gPS gene categories     | `01_supplementary_tables` | `sheets/ST4_gPS_gene_categories.csv`         |
| ST10 ChEMBL T-I pairs       | `01_supplementary_tables` | `sheets/ST5_chembl_ti_pairs.csv`             |
| ST11 drug target enrichment | `01_supplementary_tables` | `sheets/ST6_drug_target_enrichment.csv`      |
| ST12 PAV gene-disease pairs | `01_supplementary_tables` | `sheets/ST7_pav_gene_disease_pairs.csv`      |
| ST13 subgroup analysis      | `manual/build_st8.py`     | `manual/ST8_-_subgroup_analysis.xlsx`        |
| ST14 TA assignment          | `01_supplementary_tables` | `sheets/ST9_therapeutic_area_assignment.csv` |
| ST15 fine-mapping           | `04_fine_mapping_numbers` | numbers only, `manual/refreshed/`            |
| ST16 colocalisation overlap | compiled by hand          | `assets/ST16_-_coloc_overlap.xlsx`           |
| ST17 L2G performance        | `03_l2g_tables`           | `sheets/ST12_l2g_performance.csv`            |

`02_manual_table_numbers` recomputes the counts inside the hand-compiled sheets.
`sheets/ST17_ta_correlation_matrix.csv` is not a printed table; it is written by
`03-analysis-supplementary/14_genetic_correlation`.

## Checking the numbers

`tools/expected_numbers.tsv` lists every number claimed in the manuscript
Results, with the `.tex` file it came from. Each analysis notebook writes what
it computed to `results/*.json`, and `tools/check_numbers.py` compares the two
into `REPRODUCIBILITY.md` (written locally, not tracked).

The derived data are deposited in Zenodo,
[10.5281/zenodo.23736324](https://doi.org/10.5281/zenodo.23736324); the record's
`README.md` describes each dataset, and `chapters/00-data-download` fetches
them.

## Layout

```
chapters/          the pipeline, in order
src/               shared python
tools/             runners and the number checker
data/              downloaded and derived data (not tracked)
results/           numbers computed by each analysis notebook
```

## License

Apache License 2.0

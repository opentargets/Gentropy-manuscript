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
points at them.

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
| `00-data-download`          | Downloads the Open Targets 25.06 release and the project inputs (~40 GB, 30–60 min).                                       |
| `01-data-preparation`       | Thirteen notebooks, one canonical dataset each, written to `data/intermediate_files_refactor/`. Nothing else writes these. |
| `02-analysis-main`          | Six notebooks, one per Results subsection. Each writes its numbers to `results/`.                                          |
| `03-analysis-supplementary` | One notebook per Supplementary Results section.                                                                            |
| `04-figures-main`           | Figures 1–5.                                                                                                               |
| `05-figures-supplementary`  | Extended Data Figures 2–10 and the Supplementary Results figures.                                                          |
| `06-supplementary-tables`   | The supplementary table sheets.                                                                                            |
| `r-env`                     | Shared R library.                                                                                                          |
| `_legacy`                   | The pre-refactor chapters, kept for reference, not part of the pipeline.                                                   |

`src/manuscript_methods/` holds the shared code: the lead-variant-effect
pipeline, the therapeutic-area hierarchy and paths (`paper.py`), ancestry and
discovery curves (`discovery.py`), colocalisation clustering (`clusters.py`) and
drug-target enrichment statistics (`enrichment.py`).

## Checking the numbers

`tools/expected_numbers.tsv` lists every number claimed in the manuscript
Results, with the `.tex` file it came from. Each analysis notebook writes what
it computed to `results/*.json`, and `tools/check_numbers.py` compares the two
into `REPRODUCIBILITY.md`.

`GAPS.md` records what is missing: inputs that are not downloaded, manuscript
content that has no code here, and the numbers that do not reproduce, with the
reason for each.

## Layout

```
chapters/          the pipeline, in order
src/               shared python
tools/             runners and the number checker
data/              downloaded and derived data (not tracked)
results/           numbers computed by each analysis notebook
REPRODUCIBILITY.md manuscript number vs computed number
GAPS.md            what is missing and what does not reproduce
REFACTOR_PLAN.md   how the repository is organised and why
```

## License

Apache License 2.0

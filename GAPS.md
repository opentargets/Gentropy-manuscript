# Gaps — for evaluation

Three kinds of gap, all flagged rather than worked around. Nothing in this file
has been guessed at or reimplemented from the manuscript prose.

Status key: **needs a decision** = waiting on you. **blocked** = waiting on
data.

---

## 1. Data present on disk but not in the download notebook

These files are used by the pipeline and exist locally, but
`chapters/00-data-download/01_download_data_to_local_repo.ipynb` does not fetch
them, so a fresh clone cannot reproduce the analysis. They need adding to the
download notebook, or committing if small.

| File                                                                                        | Size   | Used for                                                     | Origin                                                                                                                                                                                                                                                                        |
| ------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/41586_2024_7556_MOESM8_ESM.csv`                                                       | 275 KB | Minikel et al. 2024 comparison (Results §6, SR11)            | Nature supplementary table, public                                                                                                                                                                                                                                            |
| `data/list_of_genes_32_categories.csv`                                                      | 2.2 MB | Fig 4c gene sets (a CSV duplicate of the downloaded parquet) | derived; parquet form _is_ downloaded                                                                                                                                                                                                                                         |
| `chapters/_legacy/04_making_the_list_of_gene_categories/MGIBatchReport_20251120_112357.txt` |        | mouse knockout lethal gene set (Fig 4c)                      | MGI batch query, committed in repo                                                                                                                                                                                                                                            |
| `chapters/_legacy/04_making_the_list_of_gene_categories/SourceDataFile1_FUSIL_bins.txt`     |        | FUSIL gene categories (Fig 4c)                               | published source data, committed in repo                                                                                                                                                                                                                                      |
| `chapters/_legacy/05-other-drug-indication-data/combined_ti_TA.tsv`                         |        | Pharmaprojects target–indication pairs (SR11)                | committed in repo; **licensed resource, not redistributable**                                                                                                                                                                                                                 |
| `chapters/_legacy/05-other-drug-indication-data/drug_phase_summary.tsv`                     |        | Pharmaprojects phases (SR11)                                 | as above                                                                                                                                                                                                                                                                      |
| `data/gtex_v8_ts_DEG.txt`                                                                   | 11 MB  | **nothing** — no code in the repo reads it                   | can be deleted                                                                                                                                                                                                                                                                |
| `data/l2g_training_set/20250625_gentropy_paper_v1/`                                         | 844 KB | ST12, ST13, and the Results §3 L2G model metrics             | **downloadable asset, to be added to the download notebook.** Recovered 2026-08-19 from `~/Projects/EGL_and_training_set/archive/gentropy_paper/data/`; candidate remote is `gs://genetics-portal-dev-analysis/yt4/2506_release/training_set/20250625_gentropy_paper_v1.json` |
| `data/l2g_training_set/test_v3.parquet`                                                     | 688 KB | the saved held-out split (18,611 x 30, labels + features)    | as above. The split seed was never recorded, so this saved file is the only way to reproduce the published split                                                                                                                                                              |

**The trained L2G model was never saved, and its prediction table is floored at
0.05.** `data/25.06/irene_1208_l2g_predictions` holds 1,616,456 rows with a
minimum score of 0.05, so every CS-gene pair the model scored below that is
simply absent — 16,463 of the 18,611 held-out pairs among them. Anything decided
at the L2G >= 0.5 threshold reproduces from it exactly (precision, recall, the
confusion matrix, every prioritisation in the pipeline); the two metrics that
depend on the _ranking_ of low-scoring pairs, average precision and the ROC AUC,
cannot, because those pairs collapse into a single tie. That is why S3.15/S3.16
and R3.01/R3.02 are blocked. Closing them needs the model artefact, or an
unthresholded prediction table.

### Static assets — not generated from data, shipped with the repository

These are illustrations and hand-made spreadsheets, not analysis output. They
are committed (small) and the figure scripts read them from `assets/`; a
distribution of this repository must include them or the figure cannot be
assembled.

| File                                                                           | Size   | Used for                                                                  | Origin                                                     |
| ------------------------------------------------------------------------------ | ------ | ------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `chapters/04-figures-main/figure_1/assets/Fig1 a (cropped).pdf`                | 32 KB  | **Figure 1a** — the study-flow strip across the top of Figure 1           | drawn externally; rasterised into the PDF                  |
| `chapters/04-figures-main/figure_1/assets/OT_helix_colour_RGB.png`             | 187 KB | the Open Targets helix at the centre of the Figure 1d circular Manhattan  | Open Targets brand asset                                   |
| `chapters/05-figures-supplementary/extended_data/assets/extended_figure_1.pdf` | 39 KB  | **Extended Data Fig. 1** — the data flowchart, in full                    | drawn externally; copied from the manuscript tree          |
| `chapters/06-supplementary-tables/assets/ST3_-_GSEA_results.xlsx`              | 668 KB | **Supplementary Table 3** — GSEA results, shipped as submitted            | written by hand by Polina; copied from the manuscript tree |
| `chapters/06-supplementary-tables/assets/ST11_-_coloc_overlap.xlsx`            | 8 KB   | **Supplementary Table 11** — colocalisation overlap, shipped as submitted | compiled by hand; copied from the manuscript tree          |

**Supplementary Table 11 is also the source of two Supplementary Results
numbers**, the 67% of overlaps significant by eCAVIAR and the 79% by COLOC
(S2.01, S2.02). They are its subtable 1 row totals, over an overlap set 18%
smaller than the released colocalisation tables, which give 68.0% and 79.3%.
Both are marked `precomputed` rather than failing; see
`chapters/06-supplementary-tables/manual/README.md`.

**Extended Data Fig. 1 carries about twenty computed numbers inside the
artwork**, so it goes stale silently if anything upstream changes. Every one
that can be checked was verified against the current pipeline and matches
exactly: 100,526 GWAS, 789,453 credible sets, 2,044,305 molQTL credible sets,
70,618 and 450,357 qualifying credible sets, 8,285 and 15,160 unique genes,
1,394 unique diseases, 3,412 unique measurements, 70,400 + 453,009 = 523,409
CS-gene prioritisations, 37,377 target-indication pairs and the phase counts
6,163 / 14,410 / 12,240 / 4,564. Re-check them, and redraw the figure, whenever
a qualifying definition changes.

Supplementary Methods Fig. 1 (`figure_sm1.png`, the pipeline schematic) is the
same kind of thing and has not been copied in; it lives only in the manuscript
tree.

## 2. Data missing entirely

**Closed 2026-08-19:** `data/25.06/output/target_prioritisation` has been
downloaded (1.0 MB, 78,726 targets). It was scripted in the download notebook
but the rsync had never landed. `tissueSpecificity`, `tissueDistribution` and
`hasSafetyEvent` are now available.

| #   | Input                                                                                                                                       | Blocks                                                                                                                                 | Note                                                                                                                                                 |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 4   | Enrichr `Reactome_Pathways_2024` + `KEGG_2026` (public, https://maayanlab.cloud/Enrichr)                                                    | nothing in the pipeline — **Supplementary Table 3 is out of scope**, see section 3                                                     | not a gap. Both libraries are live and their sizes (2,105 + 352 = 2,457) exactly equal ST3's row count, so the inputs are recoverable if ever needed |
| 5   | `gs://open-targets-data-releases/25.06/output/drug_molecule`                                                                                | _possibly_ the withdrawn-drug and black-box-warning gene sets in Fig 4c — being checked against `annotated_targets_wide.parquet` first | will be confirmed or dropped from this list                                                                                                          |
| 6   | Per-source study indices (`gs://finngen_data/r12/study_index`, `gs://eqtl_catalogue_data/study_index`, `gs://ukb_ppp_eur_data/study_index`) | nothing — **Supplementary Table 10 is out of scope**, see section 3                                                                    | not a gap; the published sheet was assembled by hand from numbers in several notebooks                                                               |
| 7   | `gs://open-targets-pre-data-releases/2503-testrun-1/output/l2g_feature_matrix` and the 24.09 / 24.12 `ot_genetics_portal` evidence          | the "previous Open Targets model, average precision 0.65" comparison in Results §3                                                     | pre-release buckets                                                                                                                                  |

## 3. Manuscript content with no code in this repository

| Manuscript item                                                                                                                                                               | Situation                                                                                                                                                                                                                                                                                                                                         | Needs                                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Supplementary Results 5 — importance of secondary fine-mapping signals (24,558 regions; 6,354 with primary+secondary; 2,862; **81.5 %**; 48.5/42.4 %, 9.8/7.2 %, 17.5/14.6 %) | **Closed 2026-08-20.** The code was never in this repository: it is `12_importnace_of_secondary_signals.ipynb` in `~/Projects/EGL_and_training_set/archive/gentropy_paper/`. Ported to `chapters/03-analysis-supplementary/05_secondary_signals.ipynb`, where all twelve numbers reproduce exactly, and Results 3's 81.5 % with them.             | nothing                                                                                                                                                                                                                                                                 |
| Figure 1a (circular Manhattan assembly), Extended Data Fig. 1 (flowchart), Supplementary Methods Fig. 1 (pipeline schematic)                                                  | external illustration files                                                                                                                                                                                                                                                                                                                       | nothing — will be documented as external assets                                                                                                                                                                                                                         |
| All of Supplementary Methods: Gentropy ETL, clumping, PICS/SuSiE fine-mapping, CARMA, COLOC/eCAVIAR, feature-matrix generation, L2G training run                              | upstream Open Targets pipeline; this repo consumes its released output                                                                                                                                                                                                                                                                            | nothing — documented as upstream                                                                                                                                                                                                                                        |
| Supplementary Results 14, "Estimation of the genetic correlation matrix"                                                                                                      | the LDSC run itself is external; only its output table (`canonical_pairwise_table`) is available, and it _is_ downloaded                                                                                                                                                                                                                          | nothing — the downstream analysis is reproducible                                                                                                                                                                                                                       |
| Supplementary Results 13, "Integration to Open Targets Platform"                                                                                                              | descriptive, no analysis                                                                                                                                                                                                                                                                                                                          | nothing                                                                                                                                                                                                                                                                 |
| Supplementary Table 10, fine-mapping statistics by data source                                                                                                                | **Compiled by hand** from numbers spread across several notebooks in the analysis chapters and `playground/`. Confirmed by the author 2026-08-19. There is no single script that emits this sheet                                                                                                                                                 | nothing — out of scope                                                                                                                                                                                                                                                  |
| Supplementary Table 3 (GSEA)                                                                                                                                                  | **Written by hand by Polina, not by any pipeline.** Confirmed by the author 2026-08-19; do not attempt to reproduce it. `blitzgsea` is stochastic in any case                                                                                                                                                                                     | nothing — out of scope                                                                                                                                                                                                                                                  |
| Supplementary Table 16 / therapeutic-area hierarchy                                                                                                                           | **Resolved for the gene level.** Supplementary Table 9 lists `genetic, familial or congenital disease` third, and only that order reproduces the published gene-level counts (4,743 genes in more than one area, mean 2.5308, max 21; the legacy notebook order gives 4,662 / 2.4258 / 20). The pipeline now uses the published order everywhere. | **needs a decision** — the colocalisation-cluster therapeutic-area numbers in Results 4 (4,539 clusters in more than one area, mean 1.40, max 20) were produced under the legacy order, so they will shift. The size of the shift is reported once notebook 08 has run. |
| Results §1, biobank reference lines in Fig 1c (FinnGen / MVP / UKBB gene–disease pair counts)                                                                                 | reproducible; FinnGen R12 has no publication date in the release and is assigned 2024-11-04 in code                                                                                                                                                                                                                                               | nothing — documented                                                                                                                                                                                                                                                    |

## 3b. Figures that do not reproduce the published PDF exactly

Every main figure and Extended Data figure now builds from
`data/intermediate_files_refactor`. Rendered at 1,200 px wide and compared with
the published PDFs, Figures 1 and 2 and Extended Data 2-10 are pixel-identical.
Three differences remain, none of them structural.

| Figure     | Difference | Cause                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5, panel c | 3.3 %      | The observed dashed line is the mean of 200 bootstrap lowess smooths, so it depends on which rows the resampling draws. The published run never fixed the row order of `df_for_enrichment_regression.csv`, and its order was arbitrary Spark shuffle order that cannot be recovered. The table is now sorted by `(targetId, diseaseId)`, which makes the curve reproducible from scratch but not equal to the published draw. Monte Carlo error at 200 resamples is roughly ±0.02 in probability. The solid model curves are order-independent and match exactly. |
| 4, panel b | 0.4 %      | The forest plot now reads `gene_pleiotropy_coefficients.csv` instead of re-fitting the negative-binomial models in R. The two agree to four decimals (largest relative difference 0.11 %, on missense constraint), which moves points by a pixel.                                                                                                                                                                                                                                                                                                                 |
| 3, panel c | 0.03 %     | A handful of points in the APOE scatter. Pre-existing; not investigated further.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

Extended Data Fig. 1 and Supplementary Methods Fig. 1 are external illustrations
with no source in this repository.

**Five of the six Supplementary Figures are built; SR 1 is not.**
`chapters/05-figures-supplementary/supplementary/` holds SR 2-SR 6. SR 1 is a
Weights & Biases screenshot of the L2G model evaluation, and the training set
and held-out split behind it are not available. SR 6 reproduces but needs one
input this pipeline cannot rebuild, `eit_gene_metrics-r1.csv`, from the
un-ported Supplementary Results 14.3/14.4 analysis. SR 3's two highest-MAF
points sit below the published ones, for the same reason as S6.07/S6.08. That
chapter's README records the comparison for each figure. **The asset filenames
of SR 5 and SR 6 are swapped** relative to the numbers they print as, because
the two appear in `supplementary_results.tex` in the opposite order; everything
here is named after the printed number, and `FIGURE_MAPPING.md` carries the
mapping.

Re-measured 2026-08-20, rendering both PDFs at 100 dpi, resizing to 1,200 px
wide and counting differing pixels at 2% fuzz:

| Figure                       | differing pixels  | share  |
| ---------------------------- | ----------------- | ------ |
| 1, 2, and Extended Data 2-10 | 0                 | 0.000% |
| 3                            | 202 of 618,000    | 0.033% |
| 4                            | 1,516 of 517,200  | 0.293% |
| 5                            | 22,362 of 648,000 | 3.451% |

Figure 5's share moves between runs because panel c is a fresh 200-draw
bootstrap; the rest are stable.

## 4. Analyses dropped as not present in the manuscript

Kept in `chapters/_legacy/` but not carried into the refactored pipeline,
because the manuscript text does not report them:

- `06-review-r1/ontology-duplicates/` (duplicated ontology terms; referee
  response only)
- `06-review-r1/ontology-duplicates/02_hp_mapped_gwas_review` (HP-term worklist)
- `02-analysis/01-descriptions-numbers/04_defining_novel_and_known_l2g_predictions`
- `02-analysis/06-target-enrichment/01-enrichemnt-on-target-level`,
  `06-T-I-speceficity`, `09-best_category_description`
- everything in `playground/` except the two files whose logic moved into
  `01-data-preparation/06_gene_level_table` and `08_variant_consequences`

## 5. Numbers that do not reproduce

Every table in `01-data-preparation` matches the pre-refactor tables exactly, so
where a manuscript number does not reproduce it is a question about which
quantity was measured, not about the data. Each row below says what the
refactored pipeline computes and why it differs.

Run `uv run python tools/check_numbers.py` for the current state;
`REPRODUCIBILITY.md` has the full table. As of 2026-08-20 that is **618 of the
673 registered numbers**: 130 of 153 main-text (16 mismatch, 7 blocked) and 488
of 520 supplementary (23 mismatch, 2 blocked, 7 precomputed).

**Resolved 2026-08-20 — Results 1, trait ontology terms (9,280).** Not an
ambiguity after all. The producing code is
`chapters/_legacy/02-analysis/01-descriptions-numbers/01_descriptive_numbers.ipynb`
cell 57, `si_filtered_df_one_cs.select("diseaseIds").distinct().count()` — an
array column with no `explode`, so the published value counts distinct
_trait-set combinations_ across the 39,282 studies with a credible set, not
distinct terms. That definition reproduces 9,280 exactly; the unique-term count
in the same universe is 8,159, and 12,856 over all 100,526 GWAS studies. The
author's decision is to keep 9,280 in the text rather than move numbers in a
submitted manuscript, so `02-analysis-main/01_panoramic.ipynb` now computes the
published definition and prints 8,159 beside it. Same value, same caveat, in
`supplementary_results.tex:10`, where the sentence does say "unique EFO terms".
See `chapters/02-analysis-main/README.md`.

### The text disagrees with the repository's own committed data

These reproduce identically from the pre-refactor tables and from the rebuilt
ones, so the text value appears to come from an earlier data vintage.

| Claim                                                    | Text      | Reproduced  | Evidence                                                                                                                                              |
| -------------------------------------------------------- | --------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Results 2 — non-redundant replicated CSs with PIP >= 0.5 | 120,809   | 121,490     | one row per variant and trait; identical from `variant_consequences` and from the pre-refactor `lead_variant_consequence_exploded`, both 261,334 rows |
| Results 6 — OR for Orphanet                              | 5.1       | 5.0         | the committed `drug_enrichment_subsets_vs_full_l2g.csv` says 5.0073                                                                                   |
| Results 6 — OR for OMIM                                  | 4.7       | 5.3         | the committed table says 5.3439                                                                                                                       |
| Results 6 — OR for 1 therapeutic area / >= 6             | 4.3 / 2.9 | 4.29 / 2.89 | matches `fig5b_ta_rows-r1.csv` (4.2907 / 2.8882) to four decimals; the text rounds differently                                                        |

### The measured quantity is ambiguous

| Claim                                                                                                | Text        | Reproduced                          | What is unclear                                                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------------------------------------- | ----------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Results 3 — prioritised genes with no PAV or molQTL support in 2024                                  | 26 %        | 40 %                                | 2015 reproduces exactly at 49 %, so the series is right and the endpoint definition differs                                                                                                                                                                              |
| Results 3 — nearest-gene assignments with no support                                                 | 46.1 %      | 56.8 %                              | the same assignment set reproduces eQTL 36.7 %, pQTL 5.7 % and nearest 81.2 % exactly, so "support" here must include something beyond PAV and eQTL/pQTL colocalisation                                                                                                  |
| Results 3 — assignments supported by a PAV                                                           | 13.0 %      | 12.1 %                              | 13.3 % over disease assignments only; the published sentence mixes the two sets                                                                                                                                                                                          |
| Results 4 — pleiotropic lead variants and their directionality (5,188 / 4,797 / 391 / 135 / 31 / 34) |             | 4,568 / 3,952 / 616 / 126 / 35 / 38 | computed over cluster-representative lead variants, which is the manuscript's own definition of lead_vPS. Counting every lead variant instead gives 9,000 / 7,585 / 1,415 / 186 / 59 / 47. The published values sit between the two, so a third selection rule was used. |
| Results 5 — gPS univariate beta for LoF and missense constraint                                      | 0.64 / 0.59 | 0.62 / 0.86                         | the other seven covariates of the same model reproduce; these two are the ones filled with the column mean before scaling, so the fill and scaling order matters                                                                                                         |
| Results 6 — OR for high pleiotropy against no GWAS support                                           | 0.74        | 2.97                                | an odds ratio below 1 cannot mean "more successful", so 0.74 is probably a log-odds or comes from the Figure 5c regression rather than a 2x2                                                                                                                             |
| Results 6 — OR for a previously approved target                                                      | 4.13        | 8.71                                | "previously approved for another indication" is implemented as the target having an approval other than this pair's own; the published value suggests a different reference set                                                                                          |

### Blocked on missing inputs

Nine numbers cannot be computed at all: seven main-text (R3.01, R3.02, R3.05,
R3.19, R5.23, R5.24, R5.25) and two supplementary (S3.15, S3.16). They are the
previous Open Targets model's average precision (pre-release buckets), the
loss-of-function constraint enrichment, the L2G average precision and AUC in
both places they are claimed, and the three pathway enrichment counts (Enrichr
gene sets, Supplementary Table 3 being out of scope). The L2G ranking metrics
are blocked because the trained model was never saved and the retained
prediction set covers only 11.5 % of the held-out rows — enough for precision
and recall, both of which reproduce exactly, but not for a ranking metric. See
sections 2 and 3 above.

R5.20 and R5.21 were listed here in error until 2026-08-20: `05_gene_pleiotropy`
fits all eight covariates and both match the published betas (0.5293 against
0.53, −0.1051 against −0.11). The two Supplementary Results 1 shares that needed
each source's pre-ingestion study index are now `precomputed`, not blocked.

The secondary-signal share is no longer blocked; the training set and held-out
split were recovered, which unblocked Supplementary Table 12 and five main-text
values.

## 6. Supplementary Results

`chapters/03-analysis-supplementary/` covers thirteen of the fourteen sections —
Supplementary Results 13 is descriptive and makes no quantitative claim. Of the
520 numbers registered from those sections, **487 reproduce**, 7 more are
precomputed upstream and 2 are blocked; the remaining 24 are agreed text edits,
listed section by section in that chapter's README. Two blocks are outstanding,
both with working legacy implementations to port:

- **Supplementary Results 14.3 and 14.4** — effective independent trait counts
  and disease-list subsampling, in
  `chapters/_legacy/06-review-r1/effective-independent-traits/` and
  `disease-subsampling/`.
- **Supplementary Figure SR 1** — not built; a Weights & Biases screenshot whose
  training set and held-out split are unavailable. SR 2-SR 6 are built, in
  `chapters/05-figures-supplementary/supplementary/`. SR 6 depends on
  `eit_gene_metrics-r1.csv` from the first bullet's un-ported analysis, so
  porting that analysis also closes SR 6's last gap.

`chapters/03-analysis-supplementary/README.md` lists every number that does not
reproduce, with the reason.

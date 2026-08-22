# Supplementary Results

One notebook per section of `sections/supplementary_results.tex`, in the
manuscript's own order. Each notebook writes its numbers to `results/sr*.json`;
`uv run python tools/check_numbers.py` compares them against
`tools/expected_numbers.tsv` and rewrites `REPRODUCIBILITY.md`.

A row's `status` in that TSV is `pending` (compare it), `blocked` (needs an
input this repository does not have) or `precomputed` (produced upstream of this
pipeline, before the validation it starts from, and not recoverable from the
released data). Precomputed values are also written into the notebook that would
otherwise own them, so the reason is visible where the work is.

```bash
tools/run_chapter.sh chapters/03-analysis-supplementary          # all of it
tools/run_chapter.sh chapters/03-analysis-supplementary 05 06    # selected prefixes
```

Notebooks 01-06 and 14 need Spark (40 GB driver); the rest are pandas only and
run in seconds to a few minutes. `09`, `11` and `12` are the slow pandas ones —
bootstraps, 200 held-out splits and 10,000 permutations.

| Notebook                          | Section                                           | Numbers | Reproduce                      |
| --------------------------------- | ------------------------------------------------- | ------- | ------------------------------ |
| `01_finemapping_catalogue`        | SR 1, ancestry-specific fine-mapping catalogue    | 30      | 25 + 2 precomputed             |
| `02_systematic_colocalisation`    | SR 2, systematic colocalisation                   | 9       | 7 + 2 precomputed              |
| `03_l2g_gene_prioritisation`      | SR 3, L2G gene prioritisation                     | 44      | 29 + 3 precomputed + 2 blocked |
| `04_l2g_vs_naive`                 | SR 4, L2G against naive prioritisation            | 64      | **64**                         |
| `05_secondary_signals`            | SR 5, secondary fine-mapping signals              | 12      | 11 + 1 redefined               |
| `06_variant_pleiotropy_modelling` | SR 6, variant-level pleiotropy modelling          | 17      | 14                             |
| `07_gps_and_discordance`          | SR 7, gPS against directional discordance         | 9       | **9**                          |
| `08_enrichment_bias`              | SR 8, enrichment bias from size and area          | 6       | 3                              |
| `09_nonlinearity`                 | SR 9, non-linearity of pleiotropy against success | 33      | **33**                         |
| `10_phase_transitions`            | SR 10, phase transitions by pleiotropy            | 61      | **61**                         |
| `11_criterion_validation`         | SR 11.1-11.3, the combined criterion              | 54      | **54**                         |
| `12_external_replication`         | SR 11.4-11.5, Pharmaprojects and the interaction  | 101     | 97                             |
| `13_ancestry_and_sample_size`     | SR 12, ancestry and sample size                   | 50      | **50**                         |
| `14_genetic_correlation`          | SR 14.1-14.2, the genetic correlation matrix      | 30      | **30**                         |

**487 of the 520 numbers registered from these sections reproduce**, 7 more are
precomputed upstream of this pipeline, 2 are blocked on an artefact that was
never saved, and 24 mismatch. All 24 are accounted for: 22 are mismatches **by
design** — the value computed here is the one to publish, and the registry is
held at the published number until the manuscript text is corrected, section by
section under "Manuscript text that needs updating" — and the remaining 2 are SR
6's concordance pair, deferred until that calculation is reworked.

`08_enrichment_bias.ipynb` calls R: `08_enrichment_bias.R` fits its mixed models
with `lme4::glmer` through `tools/run_r.sh`, so `chapters/r-env` has to be
restored before that notebook runs.

**SR 13, "Integration to Open Targets Platform", is descriptive and has no
notebook** — it makes no quantitative claim.

Several notebooks also supply main-text numbers that are computed nowhere else,
and register them under their `Results` identifiers: `04` gives R3.03, R3.06,
R3.07, R3.13 and R3.14; `05` gives R3.12; `13` gives R1.34, R1.35 and R1.36.
Those five Results 3 values were recorded as blocked before this chapter
existed.

## Where the code came from

Most sections had a surviving implementation. Two places to look, in this order:

1. `chapters/_legacy/` — the pre-refactor chapters, including the round-1 review
   analyses under `06-review-r1/`, which is where SR 7 through SR 12 and SR 14
   come from.
2. `~/Projects/EGL_and_training_set/archive/gentropy_paper/` — the original
   working notebooks, which is where SR 1, SR 3 and **SR 5** come from. GAPS.md
   recorded SR 5 as having no code anywhere; it is
   `12_importnace_of_secondary_signals.ipynb` there, and porting it reproduced
   all twelve of its numbers on the first run.

Shared code was factored into `src/manuscript_methods/`: `l2g.py` (the gold
standard, the prioritisation rules and their 2x2 tables, shared with
Supplementary Table 12) and `transitions.py` (phase transition rates, shared
with Extended Data Figure 8).

## What does not reproduce, and why

Ranked by how much it matters.

**Precomputed, not blocked**

- SR 1: the 27.3% of studies that fine-mapped, and the ~30% of GWAS Catalog
  studies excluded before fine-mapping. Both count the GWAS Catalog set _before_
  Open Targets ingested it, so nothing in the release can produce them — the
  denominator the 27.3% implies is about 143,890 studies against the 100,526 the
  release holds, and the release's own `qualityControls` field records
  post-ingestion reasons only (60.1% flagged, but 56,735 of the 59,048 flags are
  "harmonized summary statistics are not available", which is not an exclusion).
  Closing them would need the upstream `.../study_index/gwas_catalog`, the same
  input Supplementary Table 10's "Original number of studies before ingestion"
  column needs. Recorded in the notebook's `PRECOMPUTED` table.
- SR 2: the 67% of overlaps significant by eCAVIAR and the 79% by COLOC. The
  section makes claims on two universes — the percentages are over **all**
  tested overlaps, the credible-set counts over the **qualifying** ones — and
  the second paragraph reproduces exactly on that reading (330,584 / 63%,
  302,264 / 58%, 285,229 / 55%, 14,026 genes). The percentages, however, are the
  row totals of **Supplementary Table 11 subtable 1**, a hand-made static asset
  whose overlap universe is not the released colocalisation tables: 41,398,927 /
  61,484,864 = 67.3% and 24,536,009 / 31,167,732 = 78.7%, against 51,261,361 /
  75,407,692 = 68.0% and 30,597,242 / 38,561,709 = 79.3% in the release.
  `CLPP >= 0.01` and `CLPP > 0.01` give the identical count, so the threshold is
  not the cause, and 79% survives only because both readings round to it.
  Subtable 2 of the same sheet **does** agree with the pipeline (`All` = 285,229
  credible sets, 14,026 genes), so the sheet's two halves were built against
  different data. Recorded in the notebook's `PRECOMPUTED` table; ST11 itself is
  documented in `chapters/06-supplementary-tables/manual/README.md`.

- SR 3: the novelty comparison against Open Targets Genetics 22.10 — 456,323
  novel GWAS credible sets (58%) and 333,130 previously known. That release is
  not an input this repository downloads and nothing in 25.06 identifies which
  credible sets were already in it. Registered as S3.42-S3.44 and recorded in
  the notebook's `PRECOMPUTED` table.

**Blocked on an artefact that was never saved**

- SR 3: average precision 0.81 and AUC 0.95 on the held-out set. The trained
  model was not kept, and the saved predictions are **floored at 0.05** — every
  CS-gene pair the model scored below that has no row. That is harmless for
  anything decided at the 0.5 threshold, which is why precision 0.885 and the
  confusion matrix reproduce, but it collapses 183 of the 1,134 held-out
  positives and 16,280 of the 17,477 negatives into a single tie at the bottom
  of the ranking, and both AP and AUC are ranking metrics. Under the tie the
  area under the PR curve is 0.78 (sklearn's step-wise average precision, 0.76,
  summarises the same curve) and the ROC AUC is 0.91. The tied band is 15.0% of
  all positive-negative pairs and currently earns half credit, so the true AUC
  lies between 0.83 and 0.98 — **the published 0.95 is inside that bracket**.
  Closing these two needs the model itself, or an unthresholded prediction
  table. The same two values are already `blocked` in the main text as R3.01 and
  R3.02.

**Definitions the manuscript does not pin down**

- SR 6: the count of pleiotropic variants below full directional concordance —
  1,793 published, against 1,415 from `variant_features` and 1,260 from the
  effect matrix. No surviving notebook computes 1,793 and none of the four
  readings gives it; see the SR 6 entry below, which is open. (The disease
  count, 1,403 against the matrix's 1,308, is resolved — the published number is
  the pre-deduplication count, which is what its own parenthetical describes.)

**Approximations that are stated in the notebook**

- SR 8: **resolved** — `lme4` is now in the project R library and the two
  random-intercept models are fitted by `08_enrichment_bias.R` with `glmer`, the
  estimator the published analysis used. The fully adjusted odds ratio is now
  exactly 3.14; see the SR 8 entry below for what is left.
- SR 9 and SR 11: bootstrap and permutation summaries depend on the draw. The
  published runs fixed no seed; these notebooks do, so their numbers are stable
  but sit within Monte Carlo error of the published ones rather than on them.

**Text that disagrees with its own counts, or with the code that produced its
neighbours**

Four SR 3 numbers. In each case everything around them reproduces exactly, so
these are text corrections rather than reproduction failures; the replacement
sentences are below.

- **17,463 credible sets with more than one gene at L2G >= 0.5 (3.4%), and
  193,523 with none (37.1%).** The archive cell that produces the published
  309,989 produces this split in the same breath:
  `~/Projects/EGL_and_training_set/archive/gentropy_paper/02_descriptive_numbers_si_vi_fm.ipynb`
  cell 182 prints "met only once: 309989", "met twice: 13474", "met more than
  twice: 722", so more than one gene is 13,474 + 722 = **14,196** and the rest,
  196,790, have none. 17,463 appears in no surviving notebook, and both
  partitions sum to 520,975, so only one of the three buckets is in dispute.
- **True negatives 17,362 in the Supplementary Figure SR1 caption.** The other
  three cells reproduce exactly, and 17,362 + 95 false positives = 17,457,
  twenty short of the 17,477 held-out negatives the same section reports.
  17,382 + 95 is exact. The caption's four cells sum to 18,591 against a
  held-out set of 18,611.
- **13.0% of prioritisations supported by a PAV**, against the 63,327
  assignments the same sentence reports, which is 12.1% of 523,409. The count
  reproduces exactly, here and in
  `chapters/_legacy/03-manuscript-figures/extended_figures/ed5_l2g_venn_diagram.ipynb`;
  13.0% would need a denominator of 487,131, which appears nowhere.
- **18,950 genes (94.1%) once Orphanet, gene burden and eQTLs are added.** The
  archive's saturation notebook (`14_gene_stauration_plots.ipynb`, cells 83-88)
  unions Orphanet, OMIM, gene burden, ChEMBL, the L2G disease and measurement
  predictions, the L2G eQTL and VEP evidence and the molQTL genes and reports
  **18,809 of 20,083** protein-coding genes on valid chromosomes — 93.7%, within
  four genes of the 18,813 (93.5% of 20,130) computed here from the narrower
  union the sentence actually describes. 18,950 is in neither.

## Manuscript text that needs updating

### Supplementary Results 3

Five corrections, all where the surrounding numbers reproduce.
`tools/expected_numbers.tsv` keeps the published values, so these report as
MISMATCH by design until the text changes; update the expected values then.

**Lines 117-118** — "the model shows the precision of 0.885, selectivity of
0.994 and recall of 0.645"

> Precision 0.885 is exact. The other two are the **same values printed with a
> different rounding rule**: selectivity is 0.99456 and recall 0.6455 on the
> confusion matrix, so they round to **0.995** and **0.646**. Either correct the
> text or widen the tolerance — nothing needs recomputing.

**Line 125** (Supplementary Figure SR1 caption) — "true negatives (17,362)"

> **17,382.** False positives 95, false negatives 402 and true positives 732 are
> all exact. 17,362 + 95 = 17,457 against the section's own 17,477 held-out
> negatives; 17,382 + 95 is exact. The caption is a Weights & Biases screenshot,
> so the figure image carries the same number and would have to be regenerated,
> or the count dropped from the caption.

**Lines 129-131** — "17,463 CSs (3.4%) had more than one gene with L2G >= 0.5,
while 193,523 CSs (37.1%) had no gene"

> **14,196 CSs (2.7%)** with more than one gene and **196,790 (37.8%)** with
> none. 309,989 (59.5%) is unchanged. The archive cell that produced the
> published 309,989 prints 13,474 credible sets with two genes and 722 with more
> than two in the same output — 14,196 together.

**Lines 140-143** — "13.0% (63,327) were associated because of the
protein-altering variants (PAVs)"

> **12.1%.** The count 63,327 is unchanged, and 63,327 / 523,409 = 12.1%. Every
> other count and percentage in the sentence is exact.

**Lines 149-151** — "this results in 18,950 unique genes corresponding to 94.1%
of all protein-coding genes"

> **18,813 unique genes, 93.5%** of the 20,130 protein-coding genes in release
> 25.06, for the union the sentence describes (prioritised genes + Orphanet +
> gene burden + genes with a molQTL credible set). For reference the archive's
> wider union — adding OMIM, ChEMBL and the L2G eQTL/VEP evidence — gives 18,809
> of 20,083 genes on valid chromosomes, 93.7%. If the wider union is what was
> meant, the sentence should list those sources.

### Supplementary Results 14 — reproduces in full; two upstream definitions recovered

All 30 numbers are exact. Both discrepancies were definitions carried in
upstream artefacts rather than anything to correct in the text.

- **The disease/measurement split of the 1,114 traits comes from the upstream
  `therapeutic_area` column of `canonical_pairwise_table.parquet`**, not from
  the trait's own ontology classification. A trait is a measurement when that
  column says `measurement`; the 283 traits it does not label count as diseases.
  That gives 551 / 563 exactly. The ontology route —
  `primaryTherapeuticArea == EFO_0001444`, or the term's own `ancestors` or
  `therapeuticAreas` fields, or requiring that no disease area apply — all give
  532 / 582, because no trait in the matrix descends from both the measurement
  root and a disease root. The matrix itself is built from that same file, so
  its labels travel with it. Source:
  `_legacy/06-review-r1/ta-independence/01_within_vs_between_ta.ipynb` cell 3.
- **The single-area assignment uses the legacy ordering of the 23 roots**, not
  the Supplementary Table 9 ordering — `paper.THERAPEUTIC_AREAS_LEGACY`, which
  puts `genetic, familial or congenital disease` second to last rather than
  third. Under the ST9 ordering the same 400 diseases spread over 22 areas
  instead of 21 and 55 pairs move from within-area to between-area, shifting
  every statistic in the third decimal. This is the same two-orderings split the
  rest of the work carries: the gene-level analyses use the published order, the
  variant, cluster and genetic-correlation analyses the legacy one.

- **The coverage table's gene-measurement row** is over the genes of
  `gene_table`, the protein-coding set every gene-level analysis here uses:
  115,017 associations, 86,522 in the matrix, 75.2%. The unrestricted
  prioritisation table holds 150,360. The disease row is unaffected because all
  of its genes are already in that table, and the _term_ counts stay
  unrestricted, which is what the published table's own definition says.

### Supplementary Results 12 — reproduces; two numbers pass only on tolerance

All 50 numbers pass and the definitions match the text (5,349 disease studies in
1,525 clusters, Cameron-Trivedi dispersion held fixed, cluster-robust errors,
IRR 1.69 / 1.67 / 0.97 / 3.67 / 0.84, leave-one-cohort-out 1.46 to 1.72, FinnGen
1.46). Two of them only pass because their tolerance is loose, and both are
text-level:

- **Line 886** — "European studies were 76.9\% of GWAS published up to 2017 and
  63.8\% of those from 2018 onwards". 76.9% is exact; the second is **64.5%**.
- **Line 882** — "over-dispersion was severe (variance/mean $\approx$ 30)". The
  ratio here is **21**. The claim is qualitative and still holds, but the figure
  should match.

### Supplementary Results 11 — both mismatches explained; four numbers stay by design

**Recovered:** the Pharmaprojects genetic-support P value at line 646. The
published $1.8 \times 10^{-20}$ is the **Wald test on the log odds ratio**, the
P that belongs with the Woolf interval quoted beside it; Fisher's exact on the
same table gives 1.4e-18. `enrichment.or_rs` now returns both, and S11.72
registers the Wald one and passes.

**Lines 705-706** (the ceiling table) — the four PAV odds ratios

> The published table uses **two different reference groups in different rows**,
> and both are reproducible. Its PAV rows compare each stratum against every
> pair _outside both strata_: 10.3199 and 3.0999 in ChEMBL, 4.5016 and 1.0863 in
> Pharmaprojects — 10.32, 3.10, 4.50 and 1.09 to the digit. Its any-support rows
> compare against the _unsupported pairs only_: 4.01, 2.97, 1.88, 1.48, also
> exact.
>
> Everything else in this section, and the stratified analysis in the main text,
> uses the unsupported pairs, so that is what all four rows are registered on.
> The recomputed PAV odds ratios are therefore **10.59** and **3.18** (ChEMBL)
> and **4.58** and **1.11** (Pharmaprojects). The choice scales both odds ratios
> in a row by the same factor, so the ratio and the difference test are
> identical either way — which is why 3.33, 4.14, 4.8e-4 and 0.0014 reproduce
> regardless. The notebook prints both references side by side.

### Supplementary Results 8 — refitted with lme4; one number still short

The two mixed models are no longer approximated. `lme4` was installed into
`chapters/r-env` (and recorded in its `renv.lock`), and `08_enrichment_bias.R`
in this directory refits all four models the way the archive's
`R_scripts/05_enrichment.R` did — `glm` for the fixed-effect pair, `glmer` for
the two with a random therapeutic-area intercept. The notebook writes the model
frame, calls the script through `tools/run_r.sh`, and reads the results back;
the old variational fit is kept beside them to show what it cost.

|                                   | published | variational | `glmer`  |
| --------------------------------- | --------- | ----------- | -------- |
| genetic support alone, every pair | 3.62      | —           | **3.62** |
| with maximum sample size          | 3.44      | —           | **3.44** |
| random therapeutic area only      | 3.32      | 3.21        | 3.22     |
| both                              | 3.14      | 3.13        | **3.14** |
| therapeutic-area variance         | 0.54      | 0.56        | 0.53     |
| therapeutic-area SD               | 0.74      | 0.75        | 0.73     |

**Lines 339-343** — "Including the random TA effect reduced the enrichment
estimate to 3.14, and the variance of the TA component was estimated at 0.54
(SD~=~0.74) ... When only the random TA effect was included, excluding sample
size, the OR decreased to 3.32"

> **Decided 2026-08-20: the text takes the recomputed values.** OR **3.22** for
> the therapeutic-area only model, variance **0.53** (SD **0.73**), and 3.14
> unchanged — all three within 0.10 of what is published, and now produced by
> the same estimator the published analysis used.
>
> 3.14 is already exact. The three small gaps are in the therapeutic-area
> mapping, not the estimator: the archive's model frame carried a
> `mappedTherapeuticAreas` factor from a table this pipeline no longer has, and
> this notebook uses `efo_therapeutic_area.primaryTherapeuticArea` instead. On
> the **legacy** hierarchy the same models give 3.28 and 3.20 — closer on the
> TA-only model, further on the adjusted one — so no single hierarchy reproduces
> both, and there is nothing further to recover.
>
> `tools/expected_numbers.tsv` keeps 3.32 / 0.54 / 0.74, so S8.03, S8.05 and
> S8.06 read MISMATCH **by design** until the manuscript is edited; update the
> expected values at the same time.
>
> Worth knowing for the narrative: the published sequence "3.62 to 3.44" mixes
> two universes. 3.62 is over all 37,377 pairs and 3.44 over the 30,068 whose
> disease has a therapeutic area; on that restricted set genetic support alone
> is already 3.48, so sample size accounts for 0.04 of the drop, not 0.18.

### Supplementary Results 7 — reproduces, but downstream of the concordance rework

All nine numbers are exact and every definition in the section matches what the
notebook does: mean and maximum aggregation over a gene's L2G-prioritised
variants (maximum discordance is the minimum concordance), variants with no
concordance treated as fully concordant, `log2(gPS)` where gPS is the gene's
distinct disease count, N = 8,285.

**The dependency worth recording:** the discordance features are built from
`variant_features.betaSignConcordance`, the same column SR 6 leaves open. All
seven model numbers — S7.02 to S7.09 — will move when that calculation is
reworked. The exposure is small but real: 1,652 of the 40,706 variants are
discordant and they touch 990 of the 8,285 genes; 237 of those variants have a
single disease and would become fully concordant under a per-disease definition,
which is the only discordance evidence 152 genes have. Re-run this notebook
after SR 6 changes and re-check the section then.

### Supplementary Results 6 — deferred

**Supplementary Figure 5 caption** — "219 distinct coordinates"

> **226.** This is the cluster scatter, which prints as **Supplementary Figure
> 5** even though its asset is `figures/figure_sr6.pdf` and its label `fig:sr6`;
> see `FIGURE_MAPPING.md`. The caption's other two counts are unchanged: 20,041
> clusters (S6.15) and 13,424 of them at one disease and one therapeutic area
> (S6.17). Only the coordinate count moves, because the cluster-level
> therapeutic-area count is now taken on the Supplementary Table 9 hierarchy
> order rather than the legacy one, and the areas per cluster shift for 996
> clusters. See `chapters/02-analysis-main/README.md`, "Results 4 — the cluster
> therapeutic-area count moved to the Supplementary Table 9 order", for the full
> before/after and the reason. The figure itself is now built by
> `chapters/05-figures-supplementary/supplementary/sr06_cluster_disease_vs_ta.ipynb`.

**Lines 222-224** — "We constructed a matrix of estimated effect sizes for
40,706 disease-associated variants and 1,403 diseases (each with at least one
associated credible set)"

> **Resolved, and 1,403 now reproduces.** It is the number of diseases with at
> least one associated qualifying credible set — the archive counts it on the
> input table, before deduplication (cell 29, 77,405 rows), which is exactly
> what the parenthetical says. The _matrix_ spans **1,308**: the first dedupe
> pass keeps one row per variant and study, so a study carrying several disease
> terms contributes only one of them, and 95 diseases never survive. Suggested
> wording: "...for 40,706 disease-associated variants and 1,403 diseases with at
> least one associated credible set; after deduplication the matrix spans 1,308
> of them".

**Lines 242-244** — "Among 9,828 pleiotropic variants, 1,793 (18\%) showed
concordance~$<$~1"

> **1,793 appears in no surviving notebook** and no reading of the data gives
> it. The four candidates, all printed in the notebook:
>
> | reading                                                                                 | below 1 | universe | %        |
> | --------------------------------------------------------------------------------------- | ------- | -------- | -------- |
> | `variant_features.betaSignConcordance`, the 9,828 pleiotropic variants (registered now) | 1,415   | 9,828    | 14.4     |
> | the same column over every variant                                                      | 1,652   | 40,706   | 4.1      |
> | matrix effects, the 9,828 pleiotropic variants                                          | 1,260   | 9,828    | 12.8     |
> | matrix effects, variants with more than one disease **in the matrix**                   | 1,260   | 6,494    | **19.4** |
>
> The last row is the one that matches the sentence's own definition. The
> section says concordance "was 1 for non-pleiotropic SNPs", which is true by
> construction only when there is one signed effect per disease — under the
> `variant_features` column, computed across a variant's credible sets, 237
> single-disease variants have discordant signs across studies. The catch is
> that the matrix reading also changes the denominator: deduplication leaves
> only 6,494 of the 9,828 pleiotropic variants with more than one disease, the
> rest having lost their extra diseases to the one-row-per-variant-and-study
> rule.
>
> **Deferred, 2026-08-20.** Left as it is for now —
> `variant_features.betaSignConcordance` stays registered, so S6.07 and S6.08
> read MISMATCH at 1,415 (14.0%) against the published 1,793 (18%). **The
> concordance calculation itself is to be reworked later**, and the choice of
> denominator should be made then rather than patched now. The four readings
> above stay in the notebook so the rework starts from them.

### Supplementary Results 5

Eleven of the twelve numbers reproduce exactly from
`12_importnace_of_secondary_signals.ipynb` in the archive. The twelfth, the
headline share, was **recomputed on purpose**: the archive computes it as a
ratio of credible sets to regions, and both the supplement and the main text
state it as a share of regions.

**Lines 212-214** — "Strikingly, 81.5\% of these regions nevertheless contained
a secondary CS with at least one significant functional genomic feature"

> **50.7%.** 1,451 of the 2,862 regions whose primary credible set carries
> nothing have a secondary credible set with at least one eQTL colocalisation,
> pQTL colocalisation or PAV.
>
> The published 81.5% is 2,333 / 2,862, where the numerator counts every
> credible set carrying evidence in those regions — primary credible sets
> belonging to _other studies_ of the same region included — and the denominator
> counts regions. Different units, so the quotient is not a share of anything
> and can exceed 100%. Neighbouring readings, all printed in the notebook: 1,490
> regions (52.1%) have _any_ credible set with a feature, and at the
> region-study granularity where "primary" is actually defined, 1,598 of 3,609
> bare pairs are rescued by a secondary (44.3%).
>
> The conclusion the sentence supports is unchanged — about half of these
> regions do gain evidence from a secondary signal — and the restriction to
> replicated credible sets still tracks it, 42.5% against 50.7%.

**`results/03_colocalisation.tex` line 16** — the same claim in the main text
(R3.12): "among regions whose primary credible set lacked any eQTL/pQTL
colocalisation or PAV, 81.5\% contained a secondary credible set with at least
one such feature"

> **50.7%.** Same change; the notebook registers R3.12 from the same value, so
> the two stay in step.

**Lines 210-211** — "Of these, 6,354 regions contained both primary and
secondary CSs (8,780 and 11,439 CSs, respectively)"

> Wording only; the counts reproduce exactly. Regions are counted as distinct
> `region` values, but credible sets are ranked within `(region, studyId)`, so
> the 8,780 primaries are one per region-study pair — 1,278 of the 6,354 regions
> carry more than one study. A region also enters the comparison if _any_ study
> in it has a secondary credible set, which pulls in single-credible-set studies
> of that region whose only credible set is then counted as a primary.
> Suggested: "...6,354 regions contained both primary and secondary CSs (8,780
> and 11,439 CSs, counted once per region and study)".

**Line 207** — the PAV inequality flagged under SR 4 below

> Wording only, and harmless here: the feature is the maximum over the
> protein-coding genes of a credible set, and `> 0.66` gives the same 17.5% and
> 14.6% as `>= 0.66`. The sentence should still get SR 4's fix, since the same
> phrasing does change SR 3's and SR 4's numbers.

### Supplementary Results 4

Every cell of the published table reproduces exactly, so neither of these
changes a number — but both are definitions the section states incorrectly.

**Line 164, repeated at line 208** — "If it had a max variant effect predictor
(VEP) score~$>$~0.66 we considered it having PAV"

> Must be **$\geq$ 0.66**. The missense consequence score is exactly 0.66 —
> 60,496 of the 523,409 prioritisations sit exactly on it — so the strict
> inequality excludes almost every protein-altering variant: `vepMaximum > 0.66`
> gives the PAV row sensitivity 0.010 and FDR 0.083 against the published 0.248
> and 0.193. The `>=` reading reproduces the published row exactly. The
> colocalisation thresholds in the same sentences are unaffected — `>` and `>=`
> give identical rows for H4 0.8 and CLPP 0.01.
>
> **This is the manuscript's only definition of a PAV, so it governs SR 3,
> Extended Data Fig. 5 and the main text too.** Those all reproduce on `>=`:
> 63,327 PAV-supported prioritisations (12.1%) and 241,404
> nearest-but-unsupported (46.1%), both exact. Under `> 0.66` they would be
> 2,831 (0.5%) and 264,830 (50.6%). So no number changes — but the sentence at
> line 164 is the one that has to be fixed for SR 3's PAV counts to be correctly
> described, since SR 3 uses the term without defining it. The same sentence is
> repeated at line 208 for the secondary-signals section.

**Line 189** (Supplementary Materials Table 1) — the row labelled `L2G > 0.005`

> Harmless but misleading. The prediction table is floored at 0.05, so **no
> gene-CS pair carries a score between 0 and 0.05** and every threshold in that
> interval selects the same pairs: 0.005, 0.05 and 0.5 are three distinct rules
> but 0.005 and 0.05 are one. The row is in effect "any gene the model scored at
> all". Relabelling it `L2G >= 0.05` would describe what was computed.

### Supplementary Results 1

Three SR 1 sentences quote numbers no surviving code produces, and whose value
depends entirely on which credible sets are meant. **The definition adopted here
is the qualifying credible sets** — the analysis set used everywhere else in
this work — and the notebook exports the alternatives beside each one
(`sr1_cs_size_regressions.csv`, `sr1_sample_size_regressions.csv`,
`sr1_multi_study_pairs.csv`, `sr1_heterogeneity.csv` in the derived data
directory).

`tools/expected_numbers.tsv` still holds the published values, so S1.16, S1.29
and S1.30 report as MISMATCH by design until the text is changed. Update the
expected values at the same time.

**Line 26-29** — "slope of linear regression of size vs. MAF = 14.0, P ≪ 1e-16"

> slope = **25.97**, P = 3.1e-119, over the 520,975 qualifying credible sets (r
> = 0.032). All 787,112 GWAS credible sets give 45.2.

**Line 30** — "a negative correlation between the sample size of the study and
the size of the CSs (P ≪ 1e-16)"

> On qualifying credible sets the sign depends on the form of the predictor:
> sample size +1.3e-06 (P = 0.005), log10 sample size **-1.17** (P = 2.0e-05),
> log10 **effective** sample size **-3.30** (P = 1.2e-55). The last is
> registered, as the only reading whose P matches the claim. Note that the rank
> correlation is **positive** on the same data (Spearman +0.05 to +0.09), so the
> negative sign is a regression coefficient on the log scale and not a monotone
> negative association. If the sentence should report a correlation rather than
> a slope, its direction is wrong.

**Lines 78-82** — "16% of lead variant-disease pairs were identified in more
than one GWAS study, of which only 15% showed significant Cochran's
heterogeneity (P < 1e-4)"

> Over the 60,229 pairs of qualifying disease credible sets, **16.9%** (10,179)
> appear in more than one GWAS study. Of those, only 2,709 carry two or more
> harmonised effect estimates — most curated GWAS Catalog associations have no
> beta or standard error — and **10.4%** of those (281) reach P < 1e-4. Over all
> 10,179 multi-study pairs instead of the testable ones the share is 2.8%. The
> denominator therefore has to be stated in the sentence.

### Supplementary Results 6 and 7 — the concordance redefinition

The full protocol, the two prerequisite checks behind it and the main-text
consequences are in `chapters/02-analysis-main/README.md` under "lead_vPS and
directional concordance redefined". What it does to this chapter:

**SR 6 adopts it, including the sign gate.** S6.07 and S6.08 read
`signedLeadVPS` / `signedLeadDirectionalConcordance` on `variant_features`. The
universe is the **5,919** lead variants with `signedLeadVPS > 1`; **all** of
them have a computable concordance, because a disease with no directional
information never enters the score, so there is no undefined subgroup and the
percentage no longer depends on which denominator is chosen. A further **5,234**
lead variants have no signed contributing credible set at all and are excluded
outright.

| id    | quantity                                      | published | ungated `leadVPS`                           | amended `signedLeadVPS`                          |
| ----- | --------------------------------------------- | --------- | ------------------------------------------- | ------------------------------------------------ |
| S6.07 | pleiotropic variants with concordance below 1 | 1,793     | 1,026                                       | **1,051**                                        |
| S6.08 | those as a share of pleiotropic variants (%)  | 18        | 18 (17.8% of 5,777 defined; 16.1% of 6,383) | **18 (17.8% of 5,919, no qualification needed)** |
| —     | universe                                      | 9,828     | 6,383 (5,777 defined, 606 undefined)        | **5,919 (all defined)**                          |
| —     | excluded, nothing contributes                 | 0         | 2,433                                       | **5,234**                                        |
| —     | concordant                                    | —         | 4,751                                       | **4,868**                                        |

S6.08 stays on the published 18% and continues to **pass**; S6.07 is still a
MISMATCH, and the published 1,793 remains unreproducible under any reading.
**Line to change:** "Among 9,828 pleiotropic variants, 1,793 (18\%) showed
concordance~$<$~1" becomes **"Among 5,919 variants with lead_vPS $>$ 1, 1,051
(18\%) showed concordance $<$ 1"**, and the definition sentence above it —
"Directional concordance \dots was 1 for non-pleiotropic SNPs" — needs the
contributing-study restriction, the signed-effect requirement and the
harmonised-beta wording added.

Everything else in SR 6 is asserted unchanged in the notebook: the effect matrix
(40,706 variants, 1,403 diseases before deduplication, 1,308 in it,
largest-$|\beta|$ per disease-variant pair kept), the modelling paragraph
S6.03-S6.06 (9,828 pleiotropic, 24%, mean 1.48, max 85 — counted over all
disease terms and deliberately ungated), and S6.09-S6.17.

**SR 7 adopted the amended concordance on 2026-08-22.**
`07_gps_and_discordance.ipynb` still fits the five models four times, but
S7.01-S7.09 now report the **amended, sign-gated** column
(`signedLeadDirectionalConcordance` with missing filled to 1, which is what the
section's own text states). The published `betaSignConcordance` column and the
first redefinition are still fitted and printed beside it so both earlier
lineages stay computable.

**Registry consequence: six ids move from PASS to MISMATCH by design** —
S7.04-S7.09. S7.01 (8,285 genes), S7.02 (gPS $\beta$ 0.29) and S7.03 (gPS P
4.9e-11, inside its 5e-12 tolerance) still pass. `tools/expected_numbers.tsv` is
untouched, so these read MISMATCH until the text changes. Overall the registry
goes 608 PASS / 49 MISMATCH to **602 PASS / 55 MISMATCH**, and those six are the
only status changes in the whole registry.

None of the nine depend on the cluster representative — the aggregation runs
over every lead variant that prioritises a gene, not over representatives — so
the 2026-08-22 change of the representative rank key to `chi2Stat` leaves them
where they are.

| id                                        | published `betaSignConcordance` | redefined, undefined dropped | redefined, undefined filled with 1 | **amended sign-gated, undefined filled with 1** |
| ----------------------------------------- | ------------------------------- | ---------------------------- | ---------------------------------- | ----------------------------------------------- |
| S7.01 genes                               | 8,285                           | 8,285                        | 8,285                              | **8,285**                                       |
| S7.02 gPS $\beta$, with mean discordance  | 0.29                            | 0.29                         | 0.29                               | **0.29**                                        |
| S7.03 gPS P, with mean discordance        | 5.4e-11                         | 4.3e-11                      | 4.1e-11                            | **4.9e-11**                                     |
| S7.04 mean discordance P, joint           | 0.83                            | 0.86                         | 0.85                               | **0.98**                                        |
| S7.05 gPS $\beta$, with max discordance   | 0.25                            | 0.27                         | 0.27                               | **0.28**                                        |
| S7.06 gPS P, with max discordance         | 5.9e-07                         | 3.3e-08                      | 3.3e-08                            | **2.7e-08**                                     |
| S7.07 max discordance P, joint            | 0.14                            | 0.49                         | 0.49                               | **0.54**                                        |
| S7.08 max discordance $\beta$, univariate | 1.82                            | 1.67                         | 1.67                               | **1.64**                                        |
| S7.09 max discordance P, univariate       | 7.8e-07                         | 4.9e-05                      | 4.9e-05                            | **7.1e-05**                                     |

Dropping the undefined variants costs 236 of the 8,285 genes their concordance
aggregate; because the gene universe is the `gene_table`, N stays at 8,285 in
every column and those genes fall back to discordance 0. The two ungated columns
differ only in the _mean_ aggregation, since filling an undefined concordance
with 1 cannot change a minimum unless every variant of the gene is undefined.

The section's conclusion is unchanged and slightly sharper under either
redefinition: gPS keeps its effect in both joint models while discordance stays
non-significant, and maximum discordance is still significant univariately. What
moves is the size of the univariate discordance effect (1.82 → 1.64) and its P
value (7.8e-07 → 7.1e-05), and the joint discordance P values, which get _less_
significant (0.83 → 0.98, 0.14 → 0.54) — that is, gating out the diseases that
carried no direction removes noise from the discordance term rather than signal.

**Supplementary Figures SR2 and SR3 were rebuilt on 2026-08-22.** The
_published_ figures are PNG screenshots, but this repository has generating
notebooks for them —
`chapters/05-figures-supplementary/supplementary/sr02_clusters_by_maf.ipynb` and
`sr03_concordance_by_maf.ipynb`, both reading `cluster_covariates`, which is the
table the representative change rewrote. SR2 was rebuilt as written; SR3 was
moved onto the current concordance protocol at the author's direction, so
`signedLeadVPS` and `signedLeadDirectionalConcordance` are now carried on
`cluster_covariates` for it (an additive change to that table — the 17
pre-existing columns are value-identical and every Results 4 number is
unchanged). SR4 and SR5 were checked and are unaffected: they read
`lead_variant_effect`, `qualifying_credible_sets` and `variant_clusters`, none
of which moved.

What changed:

- **SR2**, clusters per MAF bin: 496 / 2,372 / 1,868 / 3,529 / 3,843 / 3,867 /
  4,065 became **496 / 2,360 / 1,849 / 3,579 / 3,823 / 3,877 / 4,055** (20,039
  in bins of 20,041; the two at exactly MAF 0.5 fall outside the half-open bins,
  as they always did). The total is still 20,041 and the caption's claim — rare
  variants under-represented — is untouched. The figure counts clusters at the
  representative's MAF, so **only the representative reaches it**; neither the
  first redefinition nor the sign gate does. It reproduced the published PNG
  exactly under the old representative, so this is now a deliberate departure
  from it.
- **SR3**, mean concordance per MAF bin over the pleiotropic clusters, **rebuilt
  on the current protocol**: `signedLeadDirectionalConcordance` over the
  `signedLeadVPS > 1` universe. It therefore no longer plots the published
  quantity and no pixel comparison against `figures/figure_sr3.png` is
  meaningful. Its caption still defines the quantity as the "proportion of
  same-direction associations across all trait pairs for a given variant", which
  is the superseded definition and needs replacing. All three readings, per bin
  (count, mean), printed in the notebook so the choice stays visible:

  | MAF bin   | published (3,983 universe / 3,457 valued) | first redefinition (2,081) | **current protocol (2,166)** |
  | --------- | ----------------------------------------- | -------------------------- | ---------------------------- |
  | 0-0.01    | 135, 0.9799                               | 105, 0.9697                | **106, 0.9699**              |
  | 0.01-0.05 | 413, 0.9787                               | 271, 0.9668                | **274, 0.9676**              |
  | 0.05-0.1  | 322, 0.9774                               | 183, 0.9600                | **191, 0.9616**              |
  | 0.1-0.2   | 584, 0.9703                               | 326, 0.9494                | **338, 0.9452**              |
  | 0.2-0.3   | 613, 0.9650                               | 373, 0.9451                | **387, 0.9451**              |
  | 0.3-0.4   | 657, 0.9529                               | 380, 0.9288                | **399, 0.9304**              |
  | 0.4-0.5   | 733, 0.9393                               | 443, 0.9324                | **471, 0.9331**              |

  Under the current protocol 3,019 representatives are excluded because nothing
  contributes, so the universe is 2,166 rather than the published filter's
  3,983. **The decline with MAF is steeper over the first five bins (0.9699 to
  0.9452, against 0.9799 to 0.9703 published) but flattens at the top two**:
  0.9304 then 0.9331, so the highest-MAF bin is no longer the lowest-concordance
  one. The two point estimates sit well inside each other's 95% intervals —
  [0.915, 0.946] against [0.919, 0.947] — so this is a flattening, **not a
  significant reversal**, and the ordering of the last two bins should not be
  read as a finding either way. The sentence "The proportion of variants with
  discordant effects increased with MAF" holds in direction but should not claim
  monotonicity to the top bin.

Incidentally this identifies the published 5,188: it is the number of cluster
representatives with `uniqueDiseases > 1` under the most-diseases rule,
**without** the requirement that the variant have a concordance value at all.
Requiring one gives 4,568. The published 4,797 / 391 split of the 5,188 still
does not reconcile with either.

## Still to do

- **SR 14.3 and 14.4** — the effective-independent-trait counts and the
  disease-list subsampling. Both have working implementations under
  `chapters/_legacy/06-review-r1/effective-independent-traits/` and
  `disease-subsampling/`, including `eit_lib.meff_li_ji`; neither is ported here
  yet.
- **Supplementary Figure 1 is not built** — the L2G model evaluation, a Weights
  & Biases screenshot whose training set and held-out split are unavailable.
  Figures 2-6 are built, in `chapters/05-figures-supplementary/supplementary/`;
  Figures 2, 3 and 4 illustrate this section and Figure 6 illustrates SR 14.3,
  so it needs `eit_gene_metrics-r1.csv` from the un-ported analysis above.
  Figure 3's two highest-MAF points disagree with the published ones, for the
  same reason as S6.07 and S6.08. Note that **the asset filenames of Figures 5
  and 6 are swapped** relative to the numbers they print as; `FIGURE_MAPPING.md`
  has the mapping.
- **Extended Data Figure 8 still carries its own copy** of the phase-transition
  computation that `manuscript_methods.transitions` now holds. The two agree on
  every number; rewiring the figure to read `phase_transition_rates.csv` and
  `phase_transition_tests.csv` would remove the duplication, and the figure
  would need a pixel check afterwards.

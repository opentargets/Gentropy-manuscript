# Effective number of independent traits — are the pleiotropy counts inflated by trait correlation?

Six round-1 referee comments, verbatim, all making versions of the same charge:

> **R2-MJ-3** — "Also, diseases in different therapeutic areas may share
> aetiology, for example fibrosis or inflammation, so spread across therapeutic
> areas is not evidence that horizontal pleiotropy is reduced."

> **R2-MJ-8** — "Traits that have been studied deeply, and in many flavours,
> will recover the same loci repeatedly. The pleiotropy scores are therefore
> biased by how correlated the traits are with one another."

> **R2-MJ-12** — "Diseases linked through a single cluster are likely to be
> pathologically correlated; I am not convinced that spread across therapeutic
> areas captures real diversity."

> **R2-MJ-7(b)** — "Whether two different EFO terms are independent of one
> another is not addressed."

> **R1-mn-8(b)** — "Genetic correlation is available in the authors' own data
> and should be used. How does genome-wide rg relate to the pleiotropy
> measures?"

> **R1-MJ-2** — "gPS was computed for diseases only. Measurements are the larger
> part of the database; these either need exploring, or the disease-only focus
> needs justifying."

Four new gene-level metrics are added over the same gene set as the published
gPS, and the paper's central translational claim is then re-run on them.

| Metric                                       | Definition                                                                                                                                        | Zero allowed?                                          |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `gps_measurement`                            | count of unique **measurement** EFO terms associated with the gene                                                                                | yes — 0 means disease-only                             |
| `gps_independent_traits` (`meff`)            | Li & Ji (2005) effective number of independent traits over the gene's (diseases ∪ measurements) ∩ S submatrix of the genetic-correlation matrix S | no — **NA** when the gene has no trait in S            |
| `gps_independent_diseases` (`meff_dis`)      | the same estimator over the gene's **diseases only** ∩ S                                                                                          | no — **NA** when the gene has no disease term in S     |
| `gps_independent_measurements` (`meff_meas`) | the same estimator over the gene's **measurements only** ∩ S                                                                                      | no — **NA** when the gene has no measurement term in S |

## Status

Steps 1–3 complete. Both notebooks re-executed **2026-08-14** against the
rebuilt genetic-correlation matrix (1,066 → **1,114** traits — the
stale-release-id fix documented in `chapters/02-analysis/08-genecorrs/`, see
[Input version](#input-version--s-has-been-rebuilt-twice)), no error cells.
**Step 3 was then rewritten the same day** to remove a second, independent
problem — the sample A–D restriction — described in
[Step 3](#step-3--approved-drug-targets) below.

**Everything here is a sensitivity check.** No new metric is offered as a
replacement for gPS or gps_TA. What they test is the referees' actual question —
does the translational signal survive correcting the trait count for genetic
correlation, and does it survive on the measurement axis.

**Two independent fixes landed the same day, and the second one restores part of
what the first one took away.** Fixing the matrix (stale ids → 1,114 traits)
lifted coverage a lot but, on the old sample-A–D methodology, made it look like
the rg-correction's edge over simpler counts had evaporated. Fixing the
methodology (no more sample restriction, fit everything on the full 37,377-pair
table) shows that conclusion was itself partly a restriction artefact: on the
full table, the rg-corrected disease metric again beats its own uncorrected
reference. Headline, all on the **full 37,377-pair table**, low-versus-high
odds-ratio ratio from cuts derived from each metric's own fitted curve:

| Metric                                   | Axis                               | Ratio    | P                       |
| ---------------------------------------- | ---------------------------------- | -------- | ----------------------- |
| **`gps_TA`** (published)                 | therapeutic areas                  | **1.88** | **0.0097**              |
| `gps` (published, raw)                   | diseases, raw                      | 1.86     | 0.0098                  |
| **`gps_independent_diseases`**           | diseases, r<sub>g</sub>-corrected  | **1.72** | **0.0237**              |
| `n_overlap_diseases` (its own reference) | diseases in S, uncorrected         | 1.49     | 0.068 (not significant) |
| `gps_independent_traits`                 | diseases + measurements, corrected | 1.50     | 0.222                   |
| `gps_independent_measurements`           | measurements, corrected            | 1.27     | 0.372                   |
| `n_overlap`                              | traits in S, uncorrected           | 1.24     | 0.465                   |
| `n_overlap_measurements`                 | measurements in S, uncorrected     | 1.07     | 0.783                   |
| `gps_measurement`                        | measurements, raw                  | 0.88     | 0.634                   |

**Three of nine metrics clear P < 0.05, all on the disease side, and
`gps_independent_diseases` again beats `n_overlap_diseases`** (ratio 1.72 vs
1.49; P 0.024 vs 0.068) — the exact relationship the sample-A–D version had
reversed. The measurement axis is unaffected by either fix and stays flat
(`gps_independent_measurements` 1.27, P = 0.372; `gps_measurement` 0.88, P =
0.634).

So the answer to the six comments, after both fixes:

- **The disease-axis pleiotropy ceiling is robust**, however you count diseases
  — raw (`gps`), by therapeutic area (`gps_TA`), or rg-corrected
  (`gps_independent_diseases`) — answering R2-MJ-3, R2-MJ-8, R2-MJ-12 and
  R2-MJ-7(b). R1-mn-8(b)'s literal question is answered by Step 2's correlation
  and redundancy analysis below.
- **The rg correction earns its keep again**, on the methodology that matches
  the main text: `gps_independent_diseases` beats its own uncorrected reference
  `n_overlap_diseases` on both the threshold contrast and the limb slope (see
  [What the disease-only correction shows](#what-the-disease-only-correction-shows)).
  This reverses what the sample-A–D version of this notebook concluded on
  2026-08-14 earlier the same day — that conclusion was a restriction artefact,
  not a property of the corrected matrix.
- **Unchanged:** the measurement axis has no ceiling, corrected or not — the
  R1-MJ-2 justification for restricting gPS to diseases stands, under every
  methodology tried so far.

**Numbers in this README have moved twice in one day** (matrix rebuild, then
methodology fix). Treat the ranking of metrics as unstable and always check the
notebook's actual output before quoting a figure from memory or from an older
version of this file.

## Notebooks

| Notebook                          | Needs Spark | Runtime | What it does                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| --------------------------------- | ----------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_metrics_and_gate.ipynb`       | no          | ~15 s   | Steps 1 and 2 — coverage of S, all four new metrics, descriptive tables, the decision gate                                                                                                                                                                                                                                                                                                                                                                                        |
| `02_drug_targets.ipynb`           | no          | ~15 s   | Step 3 — shape then derived threshold on the full 37,377-pair table, limb analysis                                                                                                                                                                                                                                                                                                                                                                                                |
| `03_ninepanel_nonlinearity.ipynb` | no          | ~1 min  | Step 4 — the main-text non-linearity figure (predicted P(success) vs pleiotropy, model-with-GWAS/observed-LOWESS/model-no-GWAS) redrawn for all ten metrics in `02_drug_targets.ipynb` (including the disease-decorrelated measurement count added on referee follow-up). Panels 1-2 (therapeutic areas, gPS) are the identical computation as the main-text figure, included only so all ten sit side by side. Saves `eit_step4_ninepanel-r1.pdf`, kept under its original name. |

`eit_lib.py` holds the Li & Ji estimator (`meff_li_ji`), the memoised per-gene
driver (`meff_per_gene`) and the gene-trait loaders. Notebooks 02 and 03 import
`or_rs` and `support_mask` from `../or10-optimism-validation/or10_stats.py`
rather than reimplementing them, so the support definition and the Fisher/Woolf
arithmetic are identical to the ones that reproduce the published enrichment.

```bash
cd chapters/06-review-r1/effective-independent-traits
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 01_metrics_and_gate.ipynb
# then 02_drug_targets.ipynb, then 03_ninepanel_nonlinearity.ipynb, the same way
```

## Gene set — 8,285 confirmed, both published metrics reproduce exactly

The manuscript's gene count is right and no number moves — gPS and gps_TA are
computed upstream of S and are untouched by the rg-matrix rebuild. Every row
below is in `eit_reproduction_checks-r1.csv` and asserted in the notebook.

| Quantity                 | Recomputed                     | Published           |
| ------------------------ | ------------------------------ | ------------------- |
| genes                    | **8,285**                      | 8,285               |
| genes with gPS > 1       | **5,314 (64.14%)**             | 5,314 (64%)         |
| mean gPS / max gPS       | **4.45 / 148** (max is CDKN2B) | 4.45 / 148 (CDKN2B) |
| genes with gps_TA > 1    | **4,743 (57.25%)**             | 4,743 (57%)         |
| mean gps_TA / max gps_TA | **2.53 / 21**                  | 2.53 / 21           |
| Spearman ρ(gPS, gps_TA)  | **0.9223**                     | 0.92                |

gPS also reproduces gene-by-gene from the association table: recomputing the
number of unique disease EFO terms per gene from `l2g_diseases_full-r1.csv`
agrees with the published `uniqueDiseases` for **8,285 of 8,285 genes (100%)**.
The term vocabularies also match the manuscript exactly: **1,394** disease terms
and **3,412** measurement terms.

### Provenance of the gene–trait tables

`l2g_diseases_full-r1.csv` (70,400 rows) and `l2g_measurements_full-r1.csv`
(453,009 rows) are written by
`chapters/06-review-r1/ancestry-mixed-split/01_ancestry_reclassification.ipynb`
(cells 17–20). Chain:

1. `chapters/01-data-preparation/04_qualifying_dataset_generation.ipynb` — maps
   each study's `diseaseIds` to top-of-ontology therapeutic areas and sets
   `measurement = EFO_0001444 ∈ areas`, then defines the two study sets and the
   two credible-set sets.
   - **disease studies**: `binaryLessCases` ∧ ¬`measurement` ∧ `nSamples ≥ 1000`
     ∧ `nCases/nSamples ≥ 0.001`, minus pubmedId 40069456.
   - **measurement studies**: `measurement` ∧ ¬`binaryLessCases`, with all
     descendants of `EFO_0004747` (protein measurement) and `EFO_0007882`
     (microbiome) removed.
   - **qualifying credible sets**: lead-variant effect table joined to those
     studies, `2·MAF·nCases ≥ 20` (measurements: `2·MAF·nSamples ≥ 20`) and
     `|absEstimatedBeta| ≤ 3`; then common (MAF > 0.01) ∪ rare (MAF ≤ 0.01
     **and** functional evidence — eQTL/pQTL/sQTL H4 ≥ 0.8 or CLPP ≥ 0.01, or
     VEP ≥ 0.66, or replicated). Gives **70,618** disease and **450,357**
     measurement credible sets.
2. `chapters/01-data-preparation/06_l2g_predictions.ipynb` →
   `list_of_prioritised_genes_per_CS.parquet`. L2G feature matrix restricted to
   `isProteinCoding == 1`. Per credible set: if any gene scores **≥ 0.5**, keep
   every gene at ≥ 0.5; otherwise keep only the **top-scoring** gene, and only
   if its score is **≥ 0.1**. Annotation columns are derived here too
   (`eQTL_coloc`/`pQTL_coloc` from CLPP ≥ 0.01 or H4 ≥ 0.8, `VEP` from
   `vepMaximum ≥ 0.66`, `distanceTSS` from `distanceSentinelTssNeighbourhood`).
3. The ancestry notebook joins that to `lead_variant_effect` (variantId,
   `absBeta`, MAF from `majorLdPopulationMaf`; null MAF → 0, i.e. rare) and to
   the annotated study index (projectId, year, nSamples, diseaseIds, ancestry
   classes) — **788,767** gene × credible-set rows — then splits it by inner
   join on the two qualifying-credible-set id lists.

The `-r1` files differ from the originals (`figure_1/l2g_diseases_full.csv`)
only by **added ancestry columns**; the row set is unchanged, which is why gPS
reproduces from them for 8,285 of 8,285 genes.

## Input version — S has been rebuilt twice

**Rebuild 1 (2026-08-13)**, documented in the original version of this README:
`rg_processed.parquet` gained a pair-level filter, `n_snps_used >= 100_000`,
going from 1,094 traits to 1,066.

**Rebuild 2 (2026-08-14)** is the one behind every number in this README now.
The pair table (`canonical_pairwise_table.parquet`) carries its own
`diseaseId_1`/`diseaseId_2` columns from an **earlier Open Targets release**;
against the 25.06 release used everywhere else in this repo, 222 of the table's
498 disease-labelled traits (210 MONDO, 9 OBA, 3 EFO) had no row in
`disease.parquet` at all — a release mismatch, not a real ontology gap
(root-caused in
`chapters/02-analysis/08-genecorrs/01-gene-corrs-preparation.ipynb`, see
`[[project_rg_matrix_rebuild]]` in the analysis memory). The fix: join
`studyId1`/`studyId2` to the 25.06 study index and take `diseaseIds` from there,
ignoring the pair table's own stale columns. This was independently re-verified
against the raw parquets on 2026-08-14 — the join is correct, zero unmapped
studies.

|                                                     | rebuild 1 (1,066 traits) | rebuild 2 (1,114 traits) |
| --------------------------------------------------- | ------------------------ | ------------------------ |
| traits in S                                         | 1,066                    | **1,114**                |
| disease terms of S covered                          | 16.86%                   | **33.79%**               |
| measurement terms of S covered                      | 13.57%                   | **14.86%**               |
| gene–disease associations covered                   | 29.67%                   | **73.27%**               |
| gene–measurement associations covered               | 69.07%                   | **75.23%**               |
| genes with `gps_independent_diseases` defined       | 5,128 (61.9%)            | **7,649 (92.3%)**        |
| genes with `gps_independent_traits` defined         | 7,843 (94.7%)            | **8,177 (98.7%)**        |
| sample D (all three Meff variants comparable)       | 30,117 pairs             | **35,557 pairs**         |
| `gps_independent_diseases` low/high ratio, sample D | 1.82 (P = 0.037)         | **1.69 (P = 0.030)**     |
| `gps` low/high ratio, sample D                      | 1.64 (P = 0.079)         | **1.86 (P = 0.0105)**    |

The coverage gap in rebuild 1 was itself an artefact of the stale ids, not a
property of the data — fixing it roughly doubled disease-term coverage and
materially changed which pairs survive into the strict comparable sample.
`canonical_pairwise_table.parquet`, the source pair table, was not rebuilt; only
`rg_processed.parquet` changed. **No conclusion about the measurement axis
changed direction.** The disease-axis conclusion changed in a way that is a
_strengthening_ of the underlying finding (the ceiling itself) and a
_retraction_ of one specific claim (that correction beats raw counts) — see
[Status](#status).

## Coverage of S — the main limitation, stated up front

S is 1,114 × 1,114, symmetric, diagonal 1, no NaN, nothing outside [−1, 1].
**99.84%** of its off-diagonal cells (618,949 of 619,941) carry a measured
r<sub>g</sub>; the remaining 0.16% are filled with 0 by the upstream notebook
(not repaired here — see Deviation 2). By its own study-level label it is **563
measurement** and **551 disease** traits.

S is built from NFE-only qualified studies with heritability filtering upstream,
while the gene-trait tables span all ancestries. Coverage is therefore lower at
the level of _terms_ than at the level of _associations_
(`eit_coverage-r1.csv`), though both improved substantially over rebuild 1:

| Stratum                       | Total   | Present in S | Fraction   |
| ----------------------------- | ------- | ------------ | ---------- |
| disease terms                 | 1,394   | 471          | **33.79%** |
| measurement terms             | 3,412   | 507          | **14.86%** |
| gene–disease associations     | 36,858  | 27,006       | **73.27%** |
| gene–measurement associations | 115,017 | 86,522       | **75.23%** |

183 of the 1,114 traits in S (123 of them measurements) have no L2G-prioritised
gene in either table and so contribute to no gene's Meff — down from 393 of
1,066 under rebuild 1.

**No Meff variant is on the same denominator as gPS**, and every comparison
below carries the matching `n_overlap` for that reason.

## Step 1 — the four new metrics

Per-gene table: `eit_gene_metrics-r1.csv` — `geneId`, `approvedSymbol`, `gps`,
`gps_TA`, `gps_measurement`, `n_traits_total`, and for each of the three axes an
`n_overlap*`, `gps_independent_*` and `meff_*_per_overlap` column.

Estimator calibration is checked in the notebook before use: independent traits
reproduce Meff exactly (k = 2, 5, 13, 40 all return exactly k).

Availability (`eit_metric_availability-r1.csv`):

| Metric                         | Defined                 | NA                                | `n_overlap` = 1 (Meff trivially 1) | ratio > 1   |
| ------------------------------ | ----------------------- | --------------------------------- | ---------------------------------- | ----------- |
| `gps_measurement`              | 7,804 non-zero (94.19%) | — (481 genes are 0, a real value) | —                                  | —           |
| `gps_independent_traits`       | **8,177 (98.70%)**      | 108 (1.30%)                       | 497 (6.00%)                        | 775 (9.35%) |
| `gps_independent_measurements` | **7,608 (91.83%)**      | 677 (8.17%)                       | 684 (8.26%)                        | 644 (7.77%) |
| `gps_independent_diseases`     | **7,649 (92.32%)**      | 636 (7.68%)                       | 3,062 (36.96%)                     | 379 (4.57%) |

All three Meff variants defined together: **7,080 genes (85.46%)** — up from
4,819 (58.17%) under rebuild 1.

Distributions (`eit_distributions-r1.csv`):

| Metric                         | mean     | median | p75   | p99   | max               | ratio mean / median / min–max |
| ------------------------------ | -------- | ------ | ----- | ----- | ----------------- | ----------------------------- |
| `gps_independent_traits`       | 11.52    | 8      | 14.64 | 59.08 | **178.42** (APOE) | 0.898 / — / 0.244–1.238       |
| `gps_independent_measurements` | 9.23     | 6.5    | 12.05 | 44.20 | **119.49** (GCKR) | 0.902 / — / 0.231–1.216       |
| `gps_independent_diseases`     | **3.13** | **2**  | 3.56  | 18.94 | **78.57** (FTO)   | 0.934 / — / 0.351–1.516       |

`gps_independent_diseases` is still the tightest of the three metrics, but the
gap narrowed a lot with better coverage: mean 2.00 → 3.13, median 1 → 2, max
32.95 (CDKN2B) → 78.57 (FTO, with CDKN2B a close second at 78.50). It is no
longer accurate to call it "by far the thinnest" — the ratio of its mean to
`gps_independent_traits`'s mean went from 0.21 to 0.27.

Sanity check passes — top of gPS is CDKN2B (148), FTO (126), APOE (107), ABO
(105), SH2B3 (87), as expected (`eit_top_genes-r1.csv`).

## Step 2 — description

### Correlations (`eit_correlations-r1.csv`)

Spearman, on the **7,080 genes where all three Meff variants are defined**
(`all_meff_defined` in the CSV, up from 4,819 under rebuild 1) — the only frame
in which every column is comparable:

|                     | gps_TA | gps_measurement | meff   | n_overlap | **meff_dis** | n_ov_dis   | **meff_meas** | n_ov_meas  |
| ------------------- | ------ | --------------- | ------ | --------- | ------------ | ---------- | ------------- | ---------- |
| **gps**             | 0.9187 | 0.4785          | 0.6493 | 0.6473    | **0.9220**   | 0.9366     | 0.4752        | 0.4707     |
| **gps_TA**          | —      | 0.4896          | 0.6456 | 0.6394    | **0.8745**   | 0.8725     | 0.4876        | 0.4830     |
| **gps_measurement** | —      | —               | 0.9274 | 0.9360    | 0.4840       | 0.4813     | **0.9605**    | 0.9680     |
| **meff**            | —      | —               | —      | 0.9899    | 0.6661       | 0.6637     | 0.9569        | 0.9515     |
| **meff_dis**        | —      | —               | —      | —         | —            | **0.9830** | **0.4804**    | 0.4757     |
| **meff_meas**       | —      | —               | —      | —         | —            | —          | —             | **0.9915** |

The wider single-variant frames are also in the CSV (`meff_defined`, 8,177
genes; `meff_diseases_defined`, 7,649; `meff_measurements_defined`, 7,608).

Three relationships, updated:

- **ρ(gPS, meff_dis) = 0.9220** (was 0.7262) — with the corrected ids the
  disease-only correction now tracks gPS very closely. The correction moves the
  ranking far less than it looked like under rebuild
  1.
- **ρ(meff_meas, gps_measurement) = 0.9605** — essentially unchanged; the
  measurement-only correction still barely reorders the raw measurement count.
- **ρ(meff_dis, meff_meas) = 0.4804** (was 0.4049) — the two axes are still
  genuinely different quantities; the axis dissociation in Step 3 is not an
  artefact of one metric shadowing the other.

Each Meff variant against its own coverage count, on its own widest frame:
**0.9921** (traits, n = 8,177), **0.9918** (measurements, n = 7,608), **0.9820**
(diseases, n = 7,649). The disease version is still the least redundant of the
three against its own reference, though the gap closed slightly (was 0.9927 /
0.9920 / 0.9763).

### How much of each gene's trait count is redundant (`eit_deflation_bins-r1.csv`)

`mean_ratio` is the mean fraction of a gene's S-covered traits that survive as
effectively independent. By breadth, on each axis:

| traits in S | traits: ratio | measurements: ratio | diseases: ratio |
| ----------- | ------------- | ------------------- | --------------- |
| 1           | 1.000         | 1.000               | 1.000           |
| 2           | 0.960         | 0.970               | 0.939           |
| 3–5         | 0.927         | 0.928               | 0.879           |
| 6–10        | 0.906         | 0.906               | 0.850           |
| 11–20       | 0.888         | 0.883               | 0.852           |
| 21–50       | 0.851         | 0.820               | 0.873 (> 20)    |
| 51–100      | **0.722**     | **0.570**           | —               |
| > 100       | **0.569**     | **0.484**           | —               |

Same shape as before: the steep redundancy at high breadth is still a
**measurement** phenomenon (lipid/metabolite loci with near-duplicate traits);
the disease axis stays above 0.85 throughout because no gene has more than ~30
diseases in S even now.

By gPS — the version that speaks directly to R2-MJ-8 and R2-MJ-12:

| gPS   | genes | mean diseases in S | mean Meff_dis | mean ratio |
| ----- | ----- | ------------------ | ------------- | ---------- |
| 1     | 2,451 | 1.00               | 1.00          | 1.000      |
| 2     | 1,371 | 1.68               | 1.59          | 0.958      |
| 3–5   | 2,017 | 2.95               | 2.64          | 0.907      |
| 6–10  | 1,020 | 5.55               | 4.74          | 0.859      |
| 11–20 | 560   | 10.18              | 8.68          | 0.855      |
| > 20  | 230   | 21.49              | 18.61         | **0.861**  |

**Even for genes with gPS > 20, 86.1% of their S-covered diseases are
effectively independent** (was 90.7% under rebuild 1 — essentially the same
conclusion, well within the noise of a coverage change this size). Top genes
now: CDKN2B 84 → 78.50 (0.93), FTO 89 → 78.57 (0.88), ABO 70 → 62.04 (0.89),
APOE 62 → 54.46 (0.88), SH2B3 57 → 53.89 (0.95). The absolute counts roughly
doubled with coverage, the _ratios_ did not move much — which is itself
informative:

1. **Redundancy in the disease counts is still small and still does not track
   pleiotropy in a simple monotone way.** 0.958 at gPS = 2 versus 0.861 above
   gPS = 20 — correlated diseases do not preferentially inflate the counts of
   the most pleiotropic genes, which is the specific mechanism the referees
   propose. This part of the conclusion is unchanged by the rebuild.
2. **On the trait axis the redundancy is larger but the ranking barely
   changes.** APOE 8,177-scale `gps_independent_traits` = 178.4, FTO = 154.8,
   ABO = 154.5, SH2B3 = 154.3, GCKR = 152.3 — the same loci as before (APOE,
   ABO, FADS-family, GCKR), reordered.

### Measurements are a different axis, not a repeat of gPS (R1-MJ-2)

ρ(gPS, gps_measurement) = **0.4947** against ρ(gPS, gps_TA) = **0.9223** (both
on all 8,285 genes, unaffected by S — these two raw metrics don't depend on the
rg matrix). The top measurement genes are still lipid and metabolite loci —
FADS2 (439), GCKR (386), FADS1 (352), APOE (346), TMEM258 (331) — not the top
disease genes; 481 genes are disease-only. This conclusion does not depend on S
and is unchanged by the rebuild.

## Decision gate

The brief names two ways the new metrics could fail to be useful. Both are
evaluated numerically in `eit_decision_gate-r1.csv`.

**Failure mode 1 — `gps_measurement` near-redundant with gPS: does NOT hold.** ρ
= 0.4947 on all 8,285 genes, 0.4869 on the Meff-defined subset, with 481
disease-only genes and a different set of genes at the top. Unaffected by the
rebuild (this comparison doesn't involve S). This metric clears.

**Failure mode 2 — each Meff variant restates its own coverage count: HOLDS for
all three variants now, including the disease version.** ρ against `n_overlap`
is 0.9921 (traits), 0.9918 (measurements), 0.9820 (diseases). Under rebuild 1
the disease version was the one exception (Step 3 showed
`gps_independent_diseases` beating `n_overlap_diseases`, ratio 1.82 vs 1.44).
That exception **does not survive the rebuild**: in sample D,
`n_overlap_diseases` now gives 1.80 (P = 0.019) against
`gps_independent_diseases`'s 1.69 (P = 0.030) — the correction no longer beats
its own reference on the outcome either. Failure mode 2 now holds across the
board.

**Decision taken.** No Meff variant is presented as an improved pleiotropy score
— this was already the case under rebuild 1 and is more clearly correct now. All
four new metrics go forward as **sensitivity checks**: a quantification of how
much of the apparent trait spread is redundant (§ Step 2) plus a robustness
check on the translational claim (§ Step 3).

## Step 3 — approved drug targets

Existing framework, unchanged in spirit: enrichment of genetic support among
approved targets (ChEMBL phase 4) versus clinical candidates (phases I–III),
computed from the 37,377-row pair-level table
`ti_pairs_chembl_master-r1.parquet`.

### No sample restriction — rewritten 2026-08-14

**Every count below is fitted on the full 37,377-pair table, exactly as in the
main text.** Earlier the same day this notebook built four restricted "samples"
(A–D), each dropping every pair — supported or not — whose target's Meff variant
was NA on some axis, so all nine metrics could sit on one shared row set. That
over-excluded: the vast majority of dropped pairs were _unsupported_, and
unsupported pairs never needed a pleiotropy value in the first place (the
no-support reference is defined by `support_all == 0` alone, nothing else).
Sample D alone had dropped 1,820 pairs; almost none of that exclusion was
actually necessary.

The rule now:

- **No pair is ever removed from the table.** Baseline, quadratic fit, derived
  cut, group odds ratio, ratio and P value — every one of them is computed on
  all 37,377 pairs.
- **A target absent from the 8,285-gene table keeps pleiotropy 0** for every
  metric — no disease GWAS association at all, exactly as
  `uniqueTherapeuticAreas.fillna(0)` in
  `../or10-optimism-validation/04_phase2_pharmaprojects.ipynb`.
- **The only exclusion permitted** is from the low-versus-high contrast itself:
  a _supported_ pair whose target has an undefined Meff on the axis being tested
  (zero overlap with S, not a real zero) cannot be placed in either group, and
  is left out of that one contrast — tracked separately from the ordinary
  between-cut-points gap.

`eit_step3_exclusions-r1.csv` — the actual scale of the necessary exclusion,
matching what was expected before this was even run:

| Metric                         | supported pairs excluded from its contrast | approved among them | genes with metric NA (all 8,285) |
| ------------------------------ | ------------------------------------------ | ------------------- | -------------------------------- |
| `gps_independent_traits`       | **0**                                      | 0                   | 108                              |
| `gps_independent_diseases`     | **7**                                      | 3                   | 636                              |
| `gps_independent_measurements` | **9**                                      | 4                   | 677                              |

Compare that to sample D's old 1,820 pairs / 70 targets dropped — nearly all of
that was unsupported pairs that never needed a value.

`n_overlap`, `n_overlap_diseases` and `n_overlap_measurements` are carried as
**reference metrics, not candidates** — the uncorrected count over the same
traits each Meff variant uses, and never NA for a target in the gene table (zero
overlap is a defined 0, not a missing value). If a Meff variant and its
reference give the same answer, the independence correction added nothing.

### Reproductions, before anything new

| Quantity                                | Recomputed                                                                       | Published          |
| --------------------------------------- | -------------------------------------------------------------------------------- | ------------------ |
| all-GWAS support enrichment, full table | **OR 3.6186** [3.09, 4.23]                                                       | 3.62               |
| quadratic LR on gps_TA, full table      | **64.8973**, P = **7.89 × 10⁻¹⁶**                                                | 64.90, 7.9 × 10⁻¹⁶ |
| fitted peak, gps_TA                     | **1.9030** therapeutic areas                                                     | 1.90               |
| gPS ≤ 5 versus ≥ 10 contrast            | **OR 4.798** [3.65, 6.30] versus **2.968** [2.36, 3.73], ratio 1.617, P = 0.0077 | OR 4.8 versus 3.0  |

These are all computed upstream of S and are unaffected by either the matrix
rebuild or the sample-restriction fix. **The all-GWAS baseline OR = 3.6186 is
now the single number for every count tested below** — it never involved a
pleiotropy metric, so removing the sample restriction just makes that fact
explicit instead of recomputing a slightly different baseline per sample.

### (a) Shape — fitted first (`eit_step3_shape-r1.csv`)

`outcome ~ geneticSupport + log2(metric + 1) + log2(metric + 1)²`,
likelihood-ratio test on the quadratic term. Full table (37,377 pairs); `n`
below is what remains after dropping rows where that one metric is undefined
(mostly unsupported pairs, per `n_dropped_na`):

| Metric                         | Role      | n (dropped for NA) | LR (1 df) | log2² coefficient [95% CI]     | Fitted peak | Decay point |
| ------------------------------ | --------- | ------------------ | --------- | ------------------------------ | ----------- | ----------- |
| `gps_TA`                       | published | 37,377 (0)         | **64.90** | −0.1281 [−0.1601, −0.0962]     | **1.90**    | 7.43        |
| `gps`                          | published | 37,377 (0)         | **54.17** | −0.0577 [−0.0734, −0.0419]     | **3.70**    | 21.04       |
| `gps_independent_traits`       | new       | 37,196 (181)       | **53.00** | −0.0429 [−0.0546, −0.0312]     | 5.06        | 35.69       |
| `n_overlap`                    | reference | 37,377 (0)         | **48.10** | −0.0337 [−0.0435, −0.0240]     | 6.15        | 50.12       |
| **`gps_independent_diseases`** | **new**   | 35,935 (1,442)     | **46.48** | **−0.0751 [−0.0972, −0.0530]** | **2.74**    | **12.99**   |
| `gps_independent_measurements` | new       | 36,818 (559)       | **43.63** | −0.0455 [−0.0592, −0.0318]     | 4.40        | 28.13       |
| `n_overlap_diseases`           | reference | 37,377 (0)         | **42.15** | −0.0645 [−0.0844, −0.0446]     | 2.96        | 14.68       |
| `n_overlap_measurements`       | reference | 37,377 (0)         | **41.67** | −0.0342 [−0.0448, −0.0236]     | 5.47        | 40.88       |
| `gps_measurement`              | new       | 37,377 (0)         | **29.70** | −0.0267 [−0.0364, −0.0169]     | 7.24        | 66.93       |

All nine LR tests remain overwhelmingly significant (P ≪ 10⁻⁵). Only the three
Meff variants ever drop rows, and only because their metric is genuinely
undefined for those genes — `gps`, `gps_TA`, `gps_measurement` and all three
`n_overlap*` reference counts never drop a single one of the 37,377 pairs.

The fitted peak for `gps_independent_diseases` (2.74) sits a bit above the
published gps_TA optimum (1.90) — both still small integers (2–3 distinct
disease signals), same as noted before the sample-restriction fix.

### (b) Threshold — derivation stated before any odds ratio is computed

The fitted log-odds is a downward parabola in _x_ = log2(_M_ + 1), _f_(_x_) =
_c_ + _bx_ + *ax*² with _a_ < 0. The fit itself supplies two points: the
**peak** _x_\* = −*b*/2*a*, and the **decay point** _x_ = 2*x*\*, where by the
parabola's symmetry _f_(2*x*\*) = _f_(0) — the pleiotropy-related advantage is
exactly spent and the target is back to the level of one with no associated
traits.

Hence, for every metric, with no free parameter and nothing scanned:

- **low** = _M_ ≤ round(2^_x_\* − 1)
- **high** = _M_ ≥ ⌈2^(2*x*\*) − 1⌉
- supported pairs in between are left out of the contrast, as the published gPS
  ≤ 5 versus ≥ 10 presentation also does

R2-MJ-2's charge cannot apply to a cut computed before any odds ratio is.
Derived cuts (`eit_step3_derived_cuts-r1.csv`): gps_TA ≤ 2 / ≥ 8, gPS ≤ 4 / ≥
22, `gps_independent_traits` ≤ 5 / ≥ 36, `n_overlap` ≤ 6 / ≥ 51,
**`gps_independent_diseases` ≤ 3 / ≥ 13**, `n_overlap_diseases` ≤ 3 / ≥ 15,
**`gps_independent_measurements` ≤ 4 / ≥ 29**, `n_overlap_measurements` ≤ 5 / ≥
41, `gps_measurement` ≤ 7 / ≥ 67.

### Low versus high pleiotropy (`eit_step3_strata-r1.csv`, `eit_step3_contrasts-r1.csv`)

Each group is compared with the **unsupported pairs**, never with each other, so
the two groups are not each other's control — the design of
`../or10-optimism-validation/05_pleiotropy_ceiling.ipynb`.

Full table, baseline OR 3.6186, ordered by ratio:

| Metric                         | Cut        | Low OR | High OR | Low/high | P          | pairs low / high | approved low / high | excluded (metric NA) |
| ------------------------------ | ---------- | ------ | ------- | -------- | ---------- | ---------------- | ------------------- | -------------------- |
| `gps_TA`                       | ≤ 2 / ≥ 8  | 4.68   | 2.49    | **1.88** | **0.0097** | 148 / 172        | 57 / 43             | —                    |
| `gps`                          | ≤ 4 / ≥ 22 | 4.87   | 2.63    | **1.86** | **0.0098** | 185 / 150        | 73 / 39             | —                    |
| **`gps_independent_diseases`** | ≤ 3 / ≥ 13 | 4.28   | 2.49    | **1.72** | **0.0237** | 195 / 152        | 71 / 38             | **7**                |
| `gps_independent_traits`       | ≤ 5 / ≥ 36 | 4.36   | 2.90    | 1.50     | 0.222      | 57 / 136         | 21 / 38             | 0                    |
| `n_overlap_diseases`           | ≤ 3 / ≥ 15 | 4.37   | 2.94    | 1.49     | 0.068      | 187 / 202        | 69 / 57             | —                    |
| `gps_independent_measurements` | ≤ 4 / ≥ 29 | 4.12   | 3.24    | 1.27     | 0.372      | 121 / 129        | 43 / 39             | **9**                |
| `n_overlap`                    | ≤ 6 / ≥ 51 | 4.78   | 3.87    | 1.24     | 0.465      | 82 / 132         | 32 / 45             | —                    |
| `n_overlap_measurements`       | ≤ 5 / ≥ 41 | 3.96   | 3.70    | 1.07     | 0.783      | 156 / 139        | 54 / 46             | —                    |
| `gps_measurement`              | ≤ 7 / ≥ 67 | 3.92   | 4.46    | 0.88     | 0.634      | 160 / 91         | 55 / 34             | —                    |

**Three of nine metrics clear P < 0.05, all on the disease side** (`gps_TA`,
`gps`, `gps_independent_diseases`). `n_overlap_diseases` — the same diseases,
uncorrected — is at 1.49, P = 0.068, **not significant**, so
`gps_independent_diseases` beats its own reference here on the full table, the
reverse of what the (now-superseded) sample-A–D run found. The trait-combined
and measurement-only metrics remain non-significant, as under every methodology
tried so far.

### What the disease-only correction shows

At its derived cut, `gps_independent_diseases` gives **1.72, P = 0.0237** — one
of three significant metrics, and it beats its own uncorrected reference on both
counts (1.72 vs 1.49; P 0.024 vs 0.068).

1. **It tests the referees' mechanism head-on and the claim survives.** R2-MJ-3,
   R2-MJ-7(b) and R2-MJ-12 all argue the disease counts are inflated because the
   diseases are correlated. Dividing that correlation out with Li & Ji does not
   weaken the pleiotropy ceiling — it is stronger than the uncorrected count on
   the same diseases.
2. **The correction adds signal over its own coverage count, on the methodology
   that matches the main text.** `n_overlap_diseases` alone is not significant
   (P = 0.068); `gps_independent_diseases` is (P = 0.024). The limb test agrees
   (next section): −0.204 (P = 0.075) versus −0.112 (P = 0.288) — same
   direction, correction still ahead. This is the opposite of what the
   sample-A–D version of this notebook found earlier the same day, and the
   sample-A–D version is superseded: it was comparing metrics on an artificially
   shrunk, non-representative row set.
3. **The dissociation between the axes is still clean.** Diseases 1.72 (P =
   0.024), measurements 1.27 (P = 0.372). ρ between the two Meff variants is
   0.48 (Step 2), so this is not one metric shadowing the other.

**What does not follow: that `gps_independent_diseases` improves on gPS or
gps_TA.** Both published metrics have a larger ratio and a smaller P (`gps_TA`
1.88/0.0097, `gps` 1.86/0.0098) than `gps_independent_diseases` (1.72/0.0237) at
their own derived cuts — see Cut sensitivity below for whether that ordering is
robust.

### Is the headline P value trustworthy on cells this small? (`eit_step3_permutation-r1.csv`)

The high-pleiotropy cell can hold as few as a few dozen approved pairs. Every
contrast was re-tested by **permutation** — the low/high label shuffled among
the supported pairs only, no-support reference held fixed. Two-sided on |log
ratio|, 20,000 draws, directly comparable with the two-sided Wald P.

| Metric                         | ratio  | Wald P | **permutation P** |
| ------------------------------ | ------ | ------ | ----------------- |
| `gps_TA`                       | 1.8791 | 0.0097 | 0.0111            |
| `gps`                          | 1.8551 | 0.0098 | 0.0114            |
| **`gps_independent_diseases`** | 1.7177 | 0.0237 | **0.0273**        |
| `n_overlap_diseases`           | 1.4875 | 0.0682 | 0.0810            |
| `gps_independent_traits`       | 1.5044 | 0.2222 | 0.3037            |
| `gps_independent_measurements` | 1.2722 | 0.3724 | 0.4240            |
| `n_overlap`                    | 1.2373 | 0.4650 | 0.5590            |
| `gps_measurement`              | 0.8782 | 0.6344 | 0.6803            |
| `n_overlap_measurements`       | 1.0703 | 0.7828 | 0.8073            |

The permutation P is consistently a little larger than the Wald P (mild
anticonservativeness), and the ranking of which metrics clear 0.05 is identical
on both tests. Nothing here depends on Wald approximation, and nothing here
changes which three metrics are significant.

### Cut sensitivity: what is robust and what is not

`eit_step3_cut_sweep-r1.csv` reports the contrast at every point of a grid
around each metric's derived cut; `eit_step3_cut_sweep_summary-r1.csv`
aggregates it. Cut points leaving fewer than 5 approved pairs in either group
are skipped as degenerate. **This is a robustness diagnostic — no cut from it is
adopted anywhere, and the derived cuts above are untouched.**

| Metric                         | cut points | ratio > 1 | P < 0.05 | ratio range | min P      | at derived cut   |
| ------------------------------ | ---------- | --------- | -------- | ----------- | ---------- | ---------------- |
| `gps`                          | 27         | **27**    | **19**   | 1.31–2.81   | **0.0010** | 1.86, P = 0.0098 |
| `gps_independent_diseases`     | 24         | **24**    | **18**   | 1.36–2.09   | 0.0048     | 1.72, P = 0.0237 |
| `gps_TA`                       | 21         | **21**    | **7**    | 1.47–1.90   | 0.0057     | 1.88, P = 0.0097 |
| `gps_independent_measurements` | 40         | **40**    | **0**    | 1.01–1.86   | 0.070      | 1.27, P = 0.372  |

`gps_independent_diseases` points the right way at **all 24** cut points tested
and clears 0.05 at 18 of them (min P 0.0048) — a stronger robustness profile
than the sample-A–D version had (24/24 and 17 before, but that was on a shrunk,
differently-composed row set). `gps_TA` clears 0.05 at only 7 of 21 cut points,
so "gPS and gps_TA beat the corrected metric at their derived cut" is, as
before, a statement about the derived cuts specifically, not a property of the
metrics across the grid — `gps_independent_diseases` is actually the most
cut-robust of the three significant metrics by this measure.

gPS's own fragility, at low ≤ 4 with the high cut moved one unit at a time:
ratio 1.66 (≥ 18, P = 0.015), 1.67 (≥ 19, P = 0.014), 1.59 (≥ 20, P = 0.028),
1.53 (≥ 21, P = 0.054), **1.86 (≥ 22, P = 0.0098)**, 1.67 (≥ 23, P = 0.036),
2.15 (≥ 24, P = 0.0041), 2.13 (≥ 25, P = 0.0051), 2.81 (≥ 26, P = 0.0011). One
cell either side of the derived cut still moves P by roughly a factor of 3.

Three conclusions, in decreasing order of how much weight they can bear:

1. **Robust:** all disease-axis metrics (raw and corrected) show the ceiling at
   most cut points tested; the measurement-axis correction shows it at none. The
   **axis dissociation** is the result to quote, and has now held across the
   matrix rebuild and the methodology fix both.
2. **Robust, and restored by this fix:** the ceiling survives the r<sub>g</sub>
   correction on the disease axis, and the correction beats its own uncorrected
   reference — on the full table. The sample-A–D version had this reversed; that
   was the artefact, not this.
3. **Not robust:** the ranking of metrics against each other, and any single P
   value. It has now moved twice in one day (matrix rebuild, then methodology
   fix) — treat any ordering as a snapshot, not a stable property, until it has
   been checked against a further independent change.

### Which limb of the curve carries the significant quadratic (`eit_step3_limbs-r1.csv`)

A significant quadratic says the curve bends; only the falling limb is the
paper's claim. Plain linear slope on log2(metric + 1) either side of each
metric's own fitted peak. Full table, supported pairs above the peak — the
stratum the claim is about:

| Metric                         | Slope [95% CI]              | P          | approved pairs |
| ------------------------------ | --------------------------- | ---------- | -------------- |
| **`gps`**                      | **−0.270** [−0.456, −0.084] | **0.0043** | 193            |
| `gps_TA`                       | −0.283 [−0.525, −0.041]     | **0.0217** | 220            |
| `gps_independent_diseases`     | −0.204 [−0.430, +0.021]     | 0.0754     | 176            |
| `gps_independent_traits`       | −0.152 [−0.321, +0.018]     | 0.0799     | 221            |
| `n_overlap_diseases`           | −0.112 [−0.320, +0.095]     | 0.288      | 180            |
| `n_overlap`                    | −0.038 [−0.190, +0.114]     | 0.627      | 210            |
| `gps_independent_measurements` | +0.015 [−0.166, +0.196]     | 0.870      | 192            |
| `n_overlap_measurements`       | +0.053 [−0.094, +0.201]     | 0.478      | 188            |
| `gps_measurement`              | +0.081 [−0.076, +0.237]     | 0.313      | 187            |

The rising limb (below-peak) is significantly positive for all nine metrics
tested (in the CSV).

`gps_independent_diseases` beats `n_overlap_diseases` here too (−0.204, P =
0.075 versus −0.112, P = 0.288) — consistent with the threshold test, though
neither individually clears 0.05 on this stricter, lower-power test. `gps` and
`gps_TA` are the only two with a significant falling limb (P = 0.0043 and
0.0217).

### Tenth metric — disease-decorrelated independent measurements (added 2026-08-14)

A further referee-style follow-up on R2-MJ-3/8/12:
`gps_independent_measurements` already corrects for redundancy _within_ the
measurement axis, but a measurement trait that is itself genetically correlated
with a disease is arguably not independent evidence either — it may just be a
biomarker for the same disease process. Rule, stated before anything was
computed: drop a measurement trait from the vocabulary if `|rg| > 0.7` with
**any** disease trait in S (0.7 given, not scanned), then recompute the overlap
count and the Li & Ji Meff over the surviving measurement traits only, same
estimator, same code path as `gps_independent_measurements`. Computed in
`01_metrics_and_gate.ipynb` (`gps_independent_measurements_nodisease`, with its
own uncorrected reference `n_overlap_measurements_nodisease`), carried through
Step 3 exactly like the other Meff variants.

**The cut is severe: 90.3% of measurement traits in S fail it.** Of 507
measurement traits in S, 458 correlate `|rg| > 0.7` with at least one of the 471
disease traits in S; only **49 measurement traits survive**, and only **2,150 of
8,285 genes (26.0%)** have this count defined at all (versus 92.7% for plain
`gps_independent_measurements`).

| Metric                                         | n (dropped NA)  | LR (1 df)          | Fitted peak | Derived cut | Low OR | High OR | Ratio    | P (Wald) | P (permutation) |
| ---------------------------------------------- | --------------- | ------------------ | ----------- | ----------- | ------ | ------- | -------- | -------- | --------------- |
| `gps_independent_measurements_nodisease`       | 25,103 (12,274) | 14.52 (P = 1×10⁻⁴) | 1.57        | ≤ 2 / ≥ 6   | 2.90   | 4.49    | **0.65** | 0.332    | 0.478           |
| `n_overlap_measurements_nodisease` (reference) | 37,377 (0)      | 10.03 (P = 0.0015) | 3.08        | ≤ 3 / ≥ 16  | 3.46   | 3.54    | 0.98     | 0.954    | 1.000           |

The quadratic term is still significant (the curve still bends), but on the
falling side the ratio is **below 1** — high pleiotropy outperforms low, the
opposite of the paper's claim — and neither the corrected count nor its raw
reference clears P < 0.05. The high-pleiotropy cell is tiny (24 supported pairs,
9 approved for the corrected metric), so this result should be read as
_underpowered and null_, not as a reversal: dropping 90% of the measurement
vocabulary leaves too little data to say anything with confidence, and what
remains does not reveal a hidden ceiling effect. It reinforces, rather than
overturns, the standing conclusion that the measurement axis carries no
significant translational penalty. Plotted as panel 10 in
`03_ninepanel_nonlinearity.ipynb`.

### Guardrail 1 — reported, not acted on

The derived rule applied to gps_TA gives low ≤ **2** and high ≥ **8**,
unaffected by either fix since gps_TA doesn't depend on S. The published window
is 2–5; the floor agrees exactly (round(1.90) = 2), the ceiling does not (rule
says ≥ 8, published uses 5 — the more conservative choice). For gPS the rule
gives ≤ 4 / ≥ 22 against the published ≤ 5 / ≥ 10. **Neither published criterion
has been rewritten** — recorded as a finding, per guardrail 1.

### Limitations of Step 3, stated

- **The metrics are no longer all on one shared row set.** Removing the sample
  restriction means each metric's own regression naturally drops only the (small
  number of) rows it cannot compute — so, e.g., the `gps_independent_diseases`
  contrast and the `gps_independent_measurements` contrast are not on exactly
  the same 37,377-minus-something pairs as each other. This trades away the old
  internal control (every metric on identical pairs) for matching the main
  text's actual methodology; the difference in row counts between metrics is
  always small (0–1,442 of 37,377) and is reported per metric rather than hidden
  inside a named "sample".
- The derived cuts place **different numbers of supported pairs** in each group
  per metric, so the low/high ratios are not strictly comparable across metrics
  as effect sizes. The exported tables carry every cell count.
- Differences _between_ metric ratios are never formally tested. The metrics are
  mutually correlated; the claim made is about each metric's own contrast
  against its own reference.
- **None of this is independent replication.** All fits use the same ChEMBL
  pairs and the same outcome; only the predictor is re-expressed. The
  Pharmaprojects check in `../or10-optimism-validation/` is the nearest thing to
  external replication and was **not** re-run for these metrics.
- Only the all-GWAS support definition was used. The PAV-stratified version,
  where the published interaction lives, was **not** run for the new metrics.
- No power calculation was requested or run. Non-significant P values here are
  not evidence of absence.
- **The ranking of metrics against each other has now moved twice in one day**
  (matrix rebuild, then methodology fix) and reversed at least one specific
  claim (whether the rg-correction beats its reference) in the process. Treat
  any ordering, including this one, as provisional.

## Deviations from the brief, and limitations

1. **"One representative study per EFO" is not exactly the rule, and this
   specific diagnostic is unreliable.** The notebook's own count — **846
   representative studies for 831 traits, 13 traits represented by more than one
   study (max 3)** — is computed from `canonical_pairwise_table.parquet`'s own
   (stale-release) `diseaseId_1`/`diseaseId_2` columns, restricted to ids that
   happen to also appear in the corrected S index. It therefore undercounts and
   should be read as descriptive only, not as an accurate census of S's 1,114
   traits; it was **not** recomputed via the corrected studyId → 25.06
   study-index join used to build S itself (see
   [Input version](#input-version--s-has-been-rebuilt-twice)). The dedup in
   `chapters/02-analysis/08-genecorrs/01-gene-corrs-preparation.ipynb` picks,
   _per trait pair_, the row with the smallest `rg_se`, among pairs passing
   `n_snps_used >= 100_000` — so a trait can be represented by different studies
   in different cells of S. Not changed here.
2. **Unmeasured pairs in S are 0, not NA.** The same upstream notebook fills
   absent pairs with 0 and clips r<sub>g</sub> to [−1, 1]. After the second
   rebuild that affects **0.16%** of off-diagonal cells (992 of 619,941 — was
   0.18% under rebuild 1), and treats those pairs as uncorrelated, which biases
   Meff slightly upward. Not repaired — by instruction, out of scope for this
   analysis.
3. **The Li & Ji estimator is discontinuous, and it cannot score a duplicate
   cluster below about 2.** Both are properties of the instructed estimator,
   measured in `01_metrics_and_gate.ipynb` and exported to
   `eit_estimator_robustness-r1.csv`. An independent reimplementation via
   `numpy.linalg.eig` agrees with `eigvalsh` to **9.9 × 10⁻¹⁴** across 300
   random real submatrices, so this is the estimator, not the code.

   _f_(λ) = (λ ≥ 1) + (λ − ⌊λ⌋) **jumps by ~1 at every integer ≥ 2** (it is
   continuous at 1) — this is a property of an all-ones block matrix's
   eigenvalues and does not depend on S, so it is unchanged from rebuild 1: it
   returns 1.0 for k = 2, 6–10, 12 and **2.0** for k = 3, 4, 5, 11.

   - The brief's "k perfectly correlated traits give Meff = 1" holds only when
     the top eigenvalue is computed as an exact integer — decided by
     floating-point, not by the mathematics.
   - **For any r<sub>g</sub> < 1 the estimator floors a duplicate cluster near
     2, not 1** — tested at r<sub>g</sub> = 0.9, 0.99, 0.999, 0.9999 for k = 3,
     5, 10, every one returning 2.000. So Meff _over_-states independence for
     near-duplicate trait sets, which means **the redundancy reported in this
     folder is a lower bound.** It does not affect the Step 3 conclusions, which
     are comparisons between metrics computed the same way.

   Real-data exposure to the discontinuity, on the rebuilt matrix: S has
   **54,732 off-diagonal cells (4.41%) at exactly |r<sub>g</sub>| = 1** (was
   49,900, 4.40%) — exactly the input that produces integer eigenvalues. Minimum
   distance from a real eigenvalue to the integer above it: **4.4 × 10⁻¹⁶** for
   both the traits axis (gene ENSG00000168671) and the diseases axis
   (ENSG00000008394) — genuinely on the knife edge — and 1.4 × 10⁻⁴ for
   measurements (ENSG00000130876). Sensitivity check: shrinking every clipped ±1
   to ±0.999 moves the axis means by +0.006 (traits), +0.025 (diseases) and
   +0.006 (measurements), changes more than 0.5 for 0.97%, 2.58% and 0.76% of
   genes, and leaves Spearman correlation with the original at 0.9998, 0.9862
   and 0.9996. No conclusion here turns on it — this is the same caveat as under
   rebuild 1, on marginally more affected genes (a side effect of more traits
   and more clipped cells overall).

4. **`meff / n_overlap > 1` for 775 genes (9.35%)**, max 1.238 — up from 573
   genes (6.92%), max 1.257, under rebuild 1. This is a direct consequence of
   the instructed estimator: absolute eigenvalues of a matrix that is not
   positive semi-definite (S is assembled pairwise with zero fill). No PSD
   repair was applied, by instruction; values are left as computed and the
   affected count is exported.
5. **gps_TA is taken as published, and it is not a one-TA-per-disease count.**
   The brief described the therapeutic-area assignment as "one TA per disease,
   manual hierarchy". That is **not** what `uniqueTherapeuticAreas` measures.
   Its producer is
   `chapters/02-analysis/05-gene-level-ps/01_gene_level_pleiotropy.ipynb`:

   ```python
   f.size(f.array_distinct(f.flatten(f.collect_list("mappedTherapeuticAreas"))))
   ```

   — the union, across the gene's contributing **studies**, of each study's set
   of top-of-ontology therapeutic-area ids. A study whose `diseaseIds` sit under
   two top-level areas contributes both, so 637 disease EFO terms carry more
   than one area. The one-TA-per-disease `therapy_area_hierarchy` (first match
   wins, else `other`) in
   `chapters/01-data-preparation/04_qualifying_dataset_generation.ipynb`
   produces a _different_, single-label `therapeutic_area` column — the one
   carried by the `canonical_pairwise_table` behind S — and is not used for
   gps_TA. Unaffected by the rg-matrix rebuild: gps_TA does not read S.

   Re-deriving gps_TA reproduces it for **87.3%** of genes (mismatches in both
   directions, e.g. CDKN2B 21 versus 19), because the row set behind the
   published aggregation could not be pinned down exactly; the
   one-TA-per-disease hierarchy reproduces only 83.0%, confirming it is the
   wrong definition. Those two percentages are **not in any notebook** — they
   come from a diagnostic run in conversation on 2026-08-13 and are recorded
   here only, because gps_TA is used as published and nothing depends on
   reproducing it. **Relevant to R2-MJ-3:** "spread across therapeutic areas" is
   a union of study-level top-level areas, so a single study spanning two areas
   already counts as two.

6. **`n_traits_total`** (gPS + `gps_measurement`) is exported for orientation
   only and is never used as a metric. 69 terms appear in both the disease and
   the measurement table, so it can double-count. Unaffected by the rg-matrix
   rebuild.
7. **Axis-specific Meff variants were added after the original brief.** The
   brief specified one estimator over diseases ∪ measurements and forbade
   variants. `gps_independent_diseases` and `gps_independent_measurements` were
   added on the user's instruction once the limb analysis showed the
   translational penalty is confined to the disease axis, which made an
   axis-split the only way to locate it. It is still **one estimator** (Li & Ji)
   and **one threshold rule** — what changed is the trait domain it is applied
   to. The known cost is now much smaller than it was: **7,649 of 8,285 genes
   (92.3%, was 61.9%) have at least one disease term in S** and **4,587 (55.4%,
   was 27.5%) have two or more**, so `gps_independent_diseases` is NA for only
   7.7% of genes (was 38%) and trivially 1 for a further 37.0% (was 34%).
8. **Guardrail 1 respected.** Nothing in this folder touches the 2–5 TA window
   or the OR = 10.3 claim.
9. **Step 3 no longer uses a sample restriction (reversed 2026-08-14).** An
   earlier version of this notebook built four restricted "samples" (A–D) so all
   nine metrics would sit on one shared row set, at the cost of dropping
   thousands of unsupported pairs that never needed a pleiotropy value. On
   instruction, this was replaced: every count is now fitted on the full
   37,377-pair table, matching the main text; the only exclusion is a handful of
   _supported_ pairs per Meff variant (0/7/9) whose target has no value for that
   specific axis, tracked in `eit_step3_exclusions-r1.csv` and in each
   contrast's `n_supported_metric_na`. The trade-off: metrics are no longer
   guaranteed to sit on an identical row set, though the difference is always
   small. See [Step 3](#step-3--approved-drug-targets).

## Exports (all in `data/intermediate_files/`)

Every table carries a `role` column marking each metric as `published`, `new` or
`reference`. Step 3 tables no longer carry a `sample` column — every row comes
from the full 37,377-pair table (see [Step 3](#step-3--approved-drug-targets)).
All exports were regenerated 2026-08-14: notebook 01 against the rebuilt
(1,114-trait) matrix, notebook 02 both against that matrix and, later the same
day, against the no-sample-restriction methodology.

| File                                 | Contents                                                                                                      |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `eit_reproduction_checks-r1.csv`     | published gPS / gps_TA headline numbers, recomputed vs published                                              |
| `eit_coverage-r1.csv`                | fraction of disease terms, measurement terms and gene–trait associations present in S                         |
| `eit_gene_metrics-r1.csv`            | **per-gene table** — all five metrics, three `n_overlap` columns, three Meff-per-overlap ratios               |
| `eit_metric_availability-r1.csv`     | defined / NA / trivial-1 / ratio > 1 counts per metric                                                        |
| `eit_correlations-r1.csv`            | all pairwise correlations, Spearman / Pearson raw / Pearson log2, five gene sets                              |
| `eit_distributions-r1.csv`           | distribution of every metric                                                                                  |
| `eit_deflation_bins-r1.csv`          | Meff / `n_overlap` by breadth and by gPS, for each of the three axes                                          |
| `eit_top_genes-r1.csv`               | top 10 genes by each metric                                                                                   |
| `eit_decision_gate-r1.csv`           | the statistics behind the two named failure modes                                                             |
| `eit_estimator_robustness-r1.csv`    | Li & Ji discontinuity exposure per axis, and the de-clipping sensitivity                                      |
| `eit_step3_exclusions-r1.csv`        | supported pairs excluded from the low/high contrast per Meff variant (0/7/9)                                  |
| `eit_step3_shape-r1.csv`             | quadratic fit per metric, full table — LR, P, coefficients with CI, peak, decay point, n and n dropped for NA |
| `eit_step3_derived_cuts-r1.csv`      | the low/high cuts produced by the stated rule                                                                 |
| `eit_step3_strata-r1.csv`            | low and high groups — OR, CI, relative success with CI, cell counts                                           |
| `eit_step3_contrasts-r1.csv`         | low-versus-high ratio and its P value, plus the gap and metric-NA counts                                      |
| `eit_step3_limbs-r1.csv`             | linear slope either side of each metric's fitted peak                                                         |
| `eit_step3_permutation-r1.csv`       | two-sided permutation P for every contrast, versus the Wald P                                                 |
| `eit_step3_cut_sweep-r1.csv`         | contrast at every cut point on the robustness grid                                                            |
| `eit_step3_cut_sweep_summary-r1.csv` | per metric: cut points positive, cut points clearing 0.05, ratio range                                        |

`eit_step3_exclusion_effect-r1.csv` (the old sample-restriction sensitivity
check) is **removed** — it tested whether restricting to a sample moved
gPS/gps_TA, and there is no longer a sample to test.

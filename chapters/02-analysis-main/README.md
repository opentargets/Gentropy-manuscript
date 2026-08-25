# Results

One notebook per subsection of `sections/results/`, in the manuscript's own
order. Each notebook writes its numbers to `results/*.json`;
`uv run python tools/check_numbers.py` compares them against
`tools/expected_numbers.tsv` and rewrites `REPRODUCIBILITY.md`.

```bash
tools/run_chapter.sh chapters/02-analysis-main          # all of it
tools/run_chapter.sh chapters/02-analysis-main 03 04    # selected prefixes
```

| Notebook                 | Section                                       |
| ------------------------ | --------------------------------------------- |
| `01_panoramic`           | Results 1, panoramic view across 100,526 GWAS |
| `02_selective_pressures` | Results 2, selective pressures                |
| `03_colocalisation_l2g`  | Results 3, colocalisation and L2G             |
| `04_variant_pleiotropy`  | Results 4, variant-level pleiotropy           |
| `05_gene_pleiotropy`     | Results 5, gene-level pleiotropy              |
| `06_therapeutic_success` | Results 6, therapeutic success                |

Several Results numbers are computed in `chapters/03-analysis-supplementary/`
instead, because the supplementary section owns the calculation: `04` gives
R3.03, R3.06, R3.07, R3.13 and R3.14; `05` gives R3.12; `13` gives R1.34, R1.35
and R1.36.

## Main-text numbers reviewed against the manuscript

Worked one at a time with the user, in `REPRODUCIBILITY.md` order. Each entry
records the code that produced the published value and the user's ruling.

### R1.03 — 9,280 trait ontology terms — **kept, definition documented**

`sections/results/01_panoramic.tex:13`

> Each study underwent harmonisation, quality control and manual mapping to
> 9,280 trait ontology terms spanning 23 therapeutic areas (TAs)

and `sections/supplementary_results.tex:10`

> Fine-mapping resulted in 789,453 credible sets (CSs) that cover 39,282 unique
> studies (20.24\% are binary traits) and 9,280 unique EFO terms covering all 23
> therapeutic areas (TAs).

Produced by
`chapters/_legacy/02-analysis/01-descriptions-numbers/01_descriptive_numbers.ipynb`,
cell 57:

```python
efos = si_filtered_df_one_cs.select("diseaseIds").distinct().count()
print(f"N unique EFOs: {efos}")
```

`diseaseIds` is an array column and there is no `explode`, so **9,280 is the
number of distinct trait-set combinations, not the number of distinct terms**.
The universe is the 39,282 studies with at least one credible set — the same
universe as R1.02's 4,250 publications. Four candidate definitions, all
recomputed:

| definition                                                       | count     |
| ---------------------------------------------------------------- | --------- |
| distinct `diseaseIds` arrays, studies with ≥1 CS — **published** | **9,280** |
| distinct terms, studies with ≥1 CS                               | 8,159     |
| distinct terms, all 100,526 GWAS studies                         | 12,856    |
| distinct terms, qualifying studies                               | 9,147     |

**Ruling:** leave the manuscript alone — the ambiguity is real but is not worth
moving numbers in a submitted text for. `01_panoramic` therefore reproduces the
published definition exactly and R1.03 passes; it also prints the unique-term
count (8,159) beside it so the distinction is visible where the work is. The
9,280 in `supplementary_results.tex:10` is the same value and is not separately
registered.

### R5.20, R5.21 — registry status was stale

Both were marked `blocked` in `tools/expected_numbers.tsv` on the grounds that
`tissueSpecificityBinaryNormalised` was unavailable, but `05_gene_pleiotropy`
fits all eight covariates and emits both: 0.5293 against a published 0.53, and
−0.1051 against a published −0.11. Flipped to `pending`; both pass inside the
0.011 tolerance. No manuscript change.

## Manuscript text that needs updating

`tools/expected_numbers.tsv` carries two manuscript columns: `published`, the
originally submitted value, which never changes, and `expected`, the value the
current `.tex` should carry, which is what the verdict compares against. A row
where the two differ is a manuscript edit that is owed, not a reproduction
failure; `check_numbers.py` marks those rows EDIT and counts them at the top of
`REPRODUCIBILITY.md`. The numbers below are that list.

### Results 4 — the cluster therapeutic-area count moved to the Supplementary Table 9 order

Two hierarchy orders coexisted in `src/manuscript_methods/paper.py`:
`THERAPEUTIC_AREAS`, the order published as Supplementary Table 9, and a second
one, `THERAPEUTIC_AREAS_LEGACY`, which placed
`genetic, familial or congenital disease` second to last and
`immune system disease` ninth rather than eighteenth. Both orders assign each
disease term to the first area whose ontology subtree contains it, so the order
is the assignment, and carrying two of them made the same disease carry
different areas in different sections of one paper. The gene-level analysis
(gPS, ED Fig. 7, ST 8) used the published order; the cluster-level
therapeutic-area count used the second one, which is how the published numbers
were computed. `clusters.therapeutic_area_lookup()` was made to default to
`primaryTherapeuticArea` first, feeding both `cluster_table()` and
`membership_table()`; **`THERAPEUTIC_AREAS_LEGACY` was then deleted outright on
2026-08-24 and the pipeline now carries one order** — see "The hierarchy is
unified" below.

The order touches areas only, never diseases. Verified unchanged against the
pre-change run: the 20,041 clusters and their membership, the 5,595 clusters
with more than one lead variant, every disease-count statistic (6,617 with more
than one disease, range 1-120, mean 2.1415), the 42,918 ST 15 rows and their
disease ids, all four Figure 3 variance-explained figures (R4.10-R4.13), the
directionality block R4.14-R4.22, and every gPS number.

What moves — 307 of the 12,856 terms in `efo_therapeutic_area` change area
between the two orders, which changes the area count of 996 of the 20,041
clusters (859 up, 137 down):

| id    | quantity                                         | legacy (published) | Supplementary Table 9 order                |
| ----- | ------------------------------------------------ | ------------------ | ------------------------------------------ |
| R4.06 | clusters linked to multiple TAs                  | 4,539 (23%)        | **4,766 (24%)**                            |
| R4.07 | maximum TAs per cluster                          | 20                 | **19**                                     |
| R4.08 | mean TAs per cluster                             | 1.40               | **1.45**                                   |
| R4.09 | Spearman rho, diseases against TAs per cluster   | 0.81               | **0.84**                                   |
| S6.16 | distinct coordinates in Supplementary Figure 5   | 219                | **226**                                    |
| —     | ST 15 rows whose `therapeuticArea` label changes | —                  | 6,456 of 42,918 (15.0%), over 165 diseases |

The maximum falls from 20 to 19 on the same cluster in both orders — cluster 54,
114 diseases. Its only representatives of `gastrointestinal disease` and
`reproductive system or breast disease` both move under
`genetic, familial or congenital disease`, which the cluster did not otherwise
carry: two areas out, one in. Across ST 15 the same reshuffle costs
`immune system disease` 3,753 rows and gives
`genetic, familial or congenital disease` 4,695.

**Lines 11-12 of `sections/results/04_variant_pleiotropy.tex`** — "Across these
clusters, 6,617 (33\%) were linked to multiple diseases (range 1--120, mean
2.14), and 4,539 (23\%) were linked to multiple TAs (range 1--20, mean 1.40)."

> The disease half is unchanged. The TA half becomes **4,766 (24\%)**, **range
> 1--19**, **mean 1.45**.

**Line 15** — "The two counts---unique diseases and unique TAs per
cluster---were highly correlated (Spearman $\rho = 0.81$,
$P < 1 \times 10^{-16}$)"

> **$\rho = 0.84$**; the P value is unchanged (numerically 0 in double
> precision). The conclusion drawn from it — that the two counts rank clusters
> similarly — is strengthened, not weakened.

**The default reaches three more consumers than expected.**
`clusters.therapeutic_area_lookup()` is called at five places, not two:
`cluster_table()` and `membership_table()` as intended, plus ST14 and ST16 in
`chapters/06-supplementary-tables/01_supplementary_tables.ipynb` and the Figure
3c APOE tables in `04_variant_pleiotropy.ipynb`. All three moved:

- **ST16, trait distribution across therapeutic areas** — 18 of its 22 area rows
  change count (`immune system disease` 111/75 to 29/15,
  `genetic, familial or congenital disease` 8/2 to 180/121; totals hold at 2,320
  and 1,394). This sheet was verified **exact against the published workbook**
  on 2026-08-20, under the legacy assignment, so that verification no longer
  holds — even though the published sheet's own column description says the
  assignment follows the Supplementary Table 9 hierarchy, which it did not and
  this one now does. Resolved on 2026-08-24 in favour of reissuing the sheet;
  see `chapters/06-supplementary-tables/README.md`.
- **ST14, gene-disease associations with gPS** — 5,470 of 36,858 rows (14.8%,
  164 diseases) change their `therapeuticArea` label. `gPS` and
  `numberOfTherapeuticAreas` are unchanged; no published counterpart exists.
- **Figure 3c** — `variant_pleiotropy_data_exploded.csv` goes from 192 to 193
  rows, because the table is exploded on therapeutic area and one study
  (GCST90276157, Alzheimer disease with Lewy body dementia) now spans two areas
  rather than one: Lewy body dementia moves from `nervous system disease` to
  `genetic, familial or congenital disease`. Panel c colours by disease name, so
  the extra row is a duplicate marker at coordinates that already carry a point.
  `figure_3.pdf` was rebuilt later, with the lead_vPS redefinition below.

Supplementary Results 14.1-14.2 is also on the published order now; it barely
moves: within-area mean $|r_g|$ 0.401 -> 0.400, between-area 0.317 -> 0.317,
permutation $P = 10^{-4}$ either way (55 of the 79,800 disease pairs cross from
within to between). See `chapters/03-analysis-supplementary/README.md` for the
full S14 table. The gene-level one-hot TA columns behind ED Fig. 7 and ST 8 were
already on the published order and are untouched.

### The hierarchy is unified — 2026-08-24

`THERAPEUTIC_AREAS_LEGACY` is deleted, along with the
`primaryTherapeuticAreaLegacy` and `mappedTherapeuticAreasLegacy` columns of
`efo_therapeutic_area` and `study_therapeutic_areas`, and the `column` argument
of `clusters.therapeutic_area_lookup()`. Five sites read the second order and
were repointed:

| site                                                      | what it did                                                                                                                       |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `01-data-preparation/03_therapeutic_areas`                | wrote both columns and built the `paper.TA_COLUMNS` one-hots from the legacy array                                                |
| `01-data-preparation/07_variant_features` cell 2          | aliased `mappedTherapeuticAreasLegacy` to `mappedTherapeuticAreas`, so the whole variant table was legacy under a non-legacy name |
| `02-analysis-main/04_variant_pleiotropy` cell 19          | explicit `therapeutic_area_lookup("primaryTherapeuticAreaLegacy")` for the APOE signed-direction table                            |
| `06-supplementary-tables/01_supplementary_tables` cell 26 | ran the gene-level control under both orders                                                                                      |
| `src/manuscript_methods/clusters.py`                      | documented the legacy option                                                                                                      |

**No registered number moved.** R4.06-R4.09 and S6.16 were already on the
published order and are unchanged; R4.22 (APOE therapeutic areas) stays at 15,
because the count coincides across the two orders even though the area set does
not. Verified unchanged: 20,041 clusters and their partition, 5,595 with more
than one lead variant, 6,617 with more than one disease (range 1-120, mean
2.1415), the 42,918 ST 15 rows and their disease ids, R4.10-R4.13, R4.14-R4.21,
every gPS number, Results 5 and 6, ST16's 2,320 / 1,394 / 7,010 totals, ST5
(37,377 rows), ST9 (24 rows), ST7 `uniqueTherapeuticAreas`, and `plot_a.csv` /
`plot_b.csv` / `variant_pleiotropy_data_exploded*.csv`, so no figure needed
rebuilding.

What moved:

- **ST 2, `Number of associated therapeutic areas`** — 19 of the 37 rows go up
  by one (one by three, `6_90267049_G_A` 4 -> 7); the row set and every other
  column are unchanged. Column total 231 -> 252.
- **The APOE therapeutic-area direction split**, printed in Results 4 as nine
  negative against six positive areas: now **eight negative, six positive and
  one tied** over the same 15 areas. `immune system disease` (1 disease, down)
  leaves, `genetic, familial or congenital disease` (10 diseases, 5 up / 5 down,
  tied) enters, and `nervous system disease` 13 -> 11,
  `disorder of visual system` 7 -> 4, `nutritional or metabolic disease` 6 -> 3,
  `gastrointestinal disease` 4 -> 3. This number is not in
  `expected_numbers.tsv`.
- **The study-level one-hot columns** of `study_therapeutic_areas`, and with
  them the per-area count columns and `totalStudies` of `gene_table`. That is
  the intended direction: the gene-level control in `06-supplementary-tables/01`
  now reproduces all 23 area columns and `totalStudies` for **0 of 8,285
  genes**, where the legacy one-hots gave 2,757 differing.
- **The study-level control in `01-data-preparation/03` no longer reaches zero,
  and must not be made to.** The pre-refactor `gwas_w_therapeutic_areas` had
  legacy-ordered one-hots while its gene table was on the published order, so no
  single hierarchy reproduces both. Re-expressed as a report: **2,182 of 100,526
  studies differ**, led by `geneticFamilialOrCongenitalDisease` (1,600),
  `immuneSystemDisease` (993), `musculoskeletalOrConnectiveTissueDisease` (408),
  `disorderOfVisualSystem` (250) and `nervousSystemDisease` (234);
  `totalTherapeuticAreas` differs on 61.

One unrelated breakage was fixed in passing. `01-data-preparation/07` cell 16
cross-checked `variant_features` against a parquet under `chapters/_legacy/`,
which `dcdc3ce` deleted from the repository, so the chapter could not run. The
cell now skips when the reference is absent and prints its last recorded result
(40,706 of 40,706 variants matched, `maxAbsBeta` within 4.441e-16, the rest 0);
none of the columns it checks depends on the hierarchy.

### Results 4 — lead_vPS and directional concordance redefined

Five changes, applied together because they share a universe.
`tools/expected_numbers.tsv` keeps the published values, so the numbers below
report as MISMATCH by design until the text changes.

1. **Cluster representative.** The cluster is now represented by the lead
   variant of its _seed_ credible set — the member with the smallest association
   P value, ties broken by the higher lead-variant PIP.
   `clusters.load_credible_sets` already sorts on exactly that and
   `clusters.cluster` seeds each component from the first member it reaches, so
   the seed is the most significant credible set of its cluster and no new code
   was needed to find it. The previous rule picked the lead variant associated
   with the most diseases in the cluster, choosing the covariate carrier by an
   outcome correlated with vPS itself.
2. **Contributing credible sets.** `leadVPS` counts only the credible sets of
   studies mapped to **exactly one** disease term. A study carrying several
   terms gave a variant several diseases off one association and one beta, which
   made concordance 1 by construction.
3. **Direction.** Taken from the harmonised effect-allele beta with no allele
   conversion, replacing `rescaledStatistics.minorAlleleEstimatedBeta`. Where a
   disease has more than one contributing study, the most significant
   association gives its direction.
4. **No sentinels.** A variant with no contributing credible set has no
   lead_vPS, and the flags say so rather than a 0 or a 1 standing in.
5. **The sign gate** (the amendment, applied after the four above). A credible
   set contributes only if `rescaledStatistics.directionOfEffect` is
   **non-null**. That column is null exactly where only an absolute effect size
   could be obtained, which carries no directional information, so a disease
   known to the variant only through such a credible set must not enter lead_vPS
   at all. Because every counted disease is then signed by construction, the "at
   least two signed diseases" clause drops out: concordance is computable
   wherever lead_vPS is at least 1, and equals 1 where it is 1. **The three
   reported groups collapse to two** — defined, and excluded because nothing
   contributes.

The gate is not a subtraction of the previously-undefined variants. It is
applied _before_ the per-disease "most significant study wins" pick, so it also
trims counts on variants that were never in that group (five diseases, one of
them unsigned, becomes lead*vPS 4) and can even \_change a direction*, where a
disease's most significant study was the unsigned one and its next-best signed
study takes over. The whole distribution is recomputed, not adjusted.

Scope: lead_vPS only. vPS, gPS and the Supplementary Results 6 effect matrix
still count every disease term. Three column families now sit on
`variant_features`, so all three lineages stay reproducible: the published
`uniqueDiseases` / `betaSignConcordance`, the ungated `lead*` family, and the
amended `signedLead*` family (`signedLeadVPS`, `signedLeadContributingStudies`,
`signedLeadSignedDiseases`, `signedLeadUpDiseases`, `signedLeadDownDiseases`,
`signedLeadDiseaseIds`, `signedLeadTherapeuticAreas`,
`signedLeadUniqueTherapeuticAreas`, `signedLeadDirectionalConcordance`,
`signedLeadVPSDefined`). 67 columns in all; the 46 that predate this work are
byte-identical to the pre-change run, and so are the 11 of the ungated family.

#### Correction, and the rank key reverted (2026-08-22)

The `chi2Stat` experiment below was **reverted** the same day:
`load_credible_sets` is back on stored P then PIP, and the revert restores
`variant_clusters`, `cluster_covariates`, `cluster_membership`, `plot_a.csv`,
`plot_b.csv` and every sheet byte-for-byte, `cluster_id` and row order included.

**Two statements in the section below were wrong and are corrected here.**

1. `chi2Stat` is **not** simply `chi2.isf(storedP, df=1)`.
   `PValueComponents.chi2()` in `manuscript_methods/variant_statistics.py` has
   two branches: for `pValueExponent >= -300` it is
   `chi2.isf(mantissa * 10**exponent)` (verified, max deviation 1.1e-13 over
   70,466 credible sets), and for `pValueExponent < -300` it is a **linear
   approximation on the negative log**, `4.596 * neglog - 5.367` (verified
   exactly, deviation 0.0 over 152 credible sets). The second branch never
   underflows, so `chi2Stat` _does_ carry information the float64 p-value loses.
2. "110 credible sets at the floor" conflated three different things. By
   definition: `pValue == 0.0` exactly in float64 — **31** credible sets;
   `pValueExponent <= -323` (what was actually filtered) — **110**; stored
   `(mantissa, exponent) == (1.0, -323)` — **36**; `pValueExponent < -300`, the
   approximation branch — **152**. The minimum exponent in the dataset is
   **-9,237**. The 110 span **33 distinct `(mantissa, exponent)` pairs mapping
   one-to-one onto 33 distinct `chi2Stat` values**, 1,474.57 to 42,446.50 — no
   pair maps to two values, so there was never a contradiction, only a wrong
   explanation.

So `chi2Stat` really does break ties among the **31** genuinely-underflowed
credible sets, and that is what permuted the 22 `cluster_id` values. It cannot
help cluster 26, whose three most significant credible sets store `(1.0, -323)`
**identically** — `pValue = 9.881e-324`, a denormal, not zero — and therefore
map to `chi2Stat = 1479.141` identically. The conclusion stands; the reason is
narrower.

#### Is there a field that separates cluster 26's tie?

| field                                      | cluster 26's three                                                                 | verdict                                                          |
| ------------------------------------------ | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `chi2Stat`                                 | 1479.141 for all three                                                             | no                                                               |
| `rescaledStatistics.absZScore`             | 38.4596 for all three — it is exactly `sqrt(chi2Stat)`, deviation 0.0 dataset-wide | no                                                               |
| `originalBeta`                             | 0.9221 (`19_44908684_T_C`) vs 0.5501 and 0.5131 (`19_44888997_C_T`)                | separates, but betas are not comparable across traits and scales |
| **`originalBeta / originalStandardError`** | **120.49** (`19_44908684_T_C`) vs 67.05 and 35.94                                  | **separates, and favours APOE ε4 by a wide margin**              |

So a discriminator exists, and it picks the variant the earlier note wanted. Its
coverage is the problem: `originalStandardError` is null for **44,728 of 70,618
(63.3%)** qualifying credible sets — 90.5% of GCST-with-sumstats, 68.4% of
GCST-without, 100% of the 40 unknown-provenance, 0% of FinnGen R12.
`originalBeta` is itself null for 27.9% of GCST-without-sumstats. As a primary
rank key that is unusable.

**As a tie-break it is only needed inside a tie, so dataset-wide coverage is the
wrong test** — but the coverage inside ties is also thin. Stored P ties spanning
more than one lead variant occur in **29** clusters, and `|beta/se|` is
available for every tied row in only **5** of them; the other 24 still fall
through to PIP.

#### Cost of an outcome-based tie-break (reported, not adopted)

Ranking on stored P, then the lead variant's `signedLeadVPS` descending, then
PIP:

| quantity                          | current (P, PIP)                | outcome rule                                                          |
| --------------------------------- | ------------------------------- | --------------------------------------------------------------------- |
| representatives changed           | —                               | **7 of 20,041** (bounded by the 29 multi-variant P ties, as expected) |
| cluster 26 representative         | `19_44888997_C_T`               | **`19_44908684_T_C`**                                                 |
| excluded (nothing contributes)    | 3,019                           | 3,014                                                                 |
| R4.14 / R4.15 / R4.16             | 2,166 / 1,844 / 322             | **2,167 / 1,845 / 322**                                               |
| R4.17 / R4.18 / R4.19             | 67 / 18 / 21                    | **68 / 19 / 22**                                                      |
| APOE in the R4.18 set             | no                              | **yes**                                                               |
| R4.10 / R4.11 / R4.12 / R4.13 (%) | 15.073 / 16.946 / 3.827 / 0.518 | **15.069 / 17.375 / 4.702 / 0.605**                                   |
| ST2, cluster representatives      | 18 rows / 21 genes              | **19 rows / 22 genes**                                                |
| ST2, all lead variants            | 37 rows / 37 genes              | unchanged — variant-level                                             |
| SR 6 universe / below 1           | 5,919 / 1,051                   | unchanged — variant-level, not representative-level                   |

Twelve Figure 3b coefficients, all twelve move and all in the same direction
(larger), the biggest being PAV univariate +0.0446 and sample-size univariate
+0.0385; predicted power is the only one that falls, by 0.0003 univariate and
0.0083 joint. **GERP stays non-significant in the joint model** (P 0.29 → 0.18),
so the one qualitative claim the redefinition already broke does not come back.
The `|beta/se|` hybrid tie-break gives the same answer at cluster 26 and changes
8 representatives.

Nothing here is adopted; figures were not rebuilt.

#### The chi2Stat experiment, as run (superseded by the correction above)

`clusters.load_credible_sets` now ranks on `variantStatistics.chi2Stat`
**descending**, tie-broken by higher lead-variant PIP, in place of the stored
p-value ascending with the same tie-break. Chi-square is monotone in the p-value
at one degree of freedom, so the ordering is the same in principle; the intent
was to escape the stored p-value's saturation.

**chi2Stat is non-null and strictly positive for all 70,618 qualifying credible
sets** — 0 nulls in every `projectId` × `hasSumstats` stratum (FinnGen R12 /
true 15,075; GCST / false 25,054; GCST / true 30,449; GCST / null 40). No
fallback to the stored p-value is needed, and none is implemented;
`load_credible_sets` raises if a null ever appears.

**The saturation is real but chi2Stat does not escape it.** 110 credible sets
across 77 studies report the stored p-value as `1.0e-323`, the double-precision
floor. `chi2Stat` is computed _from_ that stored mantissa and exponent —
`chi2.isf(pValue, df=1)` in `manuscript_methods.variant_statistics` — so it is a
strictly monotone transform of the same saturated number, carrying exactly the
same information: 42,730 distinct `(mantissa, exponent)` pairs map to 42,730
distinct `chi2Stat` values, one to one. In cluster 26 all three floor credible
sets land on `chi2Stat = 1479.141000` exactly, so the tie survives, PIP decides
it as before, and the representative stays `19_44888997_C_T`.
**`19_44908684_T_C` does not become the representative.** Escaping the floor
would need an input that is not derived from the stored p-value — a z-score or a
beta and standard error.

What the change does do, measured against the pre-change run:

| quantity                                                                | result                                                                                                                            |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| cluster partition                                                       | **identical** — symmetric difference 0 over 20,041 clusters                                                                       |
| representatives (`leadVariantId`) that change                           | **0 of 20,041**                                                                                                                   |
| clusters whose seed _credible set_ changes, same representative variant | 1                                                                                                                                 |
| `cluster_id` values permuted                                            | 22 of 20,041                                                                                                                      |
| clusters seeded from a residual tie on both keys                        | 87, none spanning more than one lead variant, so the representative variant is never ambiguous                                    |
| every `results/*.json` value                                            | unchanged, except R4.10-R4.13 at the 1e-13 level (row-permutation noise in the model fit: 15.073413831782467 → 15.07341383178247) |
| twelve Figure 3b coefficients                                           | unchanged — asserted, max deviation below 1e-6                                                                                    |
| `figure_3.pdf`                                                          | **0 of 1,365,525 pixels differ** from the pre-change build                                                                        |
| `cluster_covariates`, `cluster_membership`                              | content identical once sorted; only `cluster_id` and row order permuted                                                           |
| ST15 sheet                                                              | changes, in the `cluster_id` column only (see `chapters/06-supplementary-tables/README.md`)                                       |

For the record, the old rule had a p-value tie spanning more than one lead
variant in **29** clusters, so PIP was deciding the representative variant in 29
places, not just cluster 26.

#### Sizing the sign gate

Over the 65,431 qualifying credible sets whose study maps to exactly one disease
term:

| quantity                                 | count                                                                                            |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `directionOfEffect` null                 | **5,706 (8.7%)**                                                                                 |
| `originalBeta` null                      | 5,705                                                                                            |
| direction null while the beta is present | **1** — `GCST90270934` at `X_15523993_G_A`, whose beta is exactly `0.0`, so a signum has no sign |
| direction present while the beta is null | 0                                                                                                |
| `directionOfEffect == 0`                 | 0 (the column is null rather than 0 by construction)                                             |

So the gate column _is_ the beta column, and the gate loses nothing an effect
size would have kept. All of it sits in one place: curated GWAS Catalog studies
without harmonised summary statistics (`projectId = GCST`,
`hasSumstats = false`) contribute 5,687 of the 5,706, and a further 19 come from
31 credible sets whose `hasSumstats` is null. GCST **with** sumstats and FinnGen
R12 contribute **zero** between them, across 42,641 credible sets.

Per lead variant, of the 38,273 with a contributing credible set:

|                            | count               |
| -------------------------- | ------------------- |
| lose every disease         | **2,801**           |
| lose some but not all      | **675**             |
| untouched                  | 34,797              |
| contributing disease terms | 50,519 → **46,816** |

#### Which field carries the harmonised signed beta

**`originalBeta`, whose sign is already stored as
`rescaledStatistics.directionOfEffect`.** `originalBeta` is the release
`credible_set.beta`, the effect of the **alternate** allele of the variant id. A
variant id is `chromosome_position_ref_alt`, so the allele the effect is
measured against is fixed by the id and is the same in every credible set of
that variant, with no conversion needed. The three alternatives:

| field                                                 | why not                                                                                                                                                                                                                                                                                          |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `rescaledStatistics.minorAlleleEstimatedBeta`         | the one previously used. It is `directionOfEffect * absEstimatedBeta`, negated when `majorLdPopulationAf.alleleFrequency > 0.5` — the alternate-allele frequency **in that study's own major LD population**. The reference allele therefore depends on the study's ancestry, not on the variant |
| `rescaledStatistics.absEstimatedBeta`, `absZScore`    | unsigned by construction                                                                                                                                                                                                                                                                         |
| `originalStandardError`, `variantStatistics.chi2Stat` | carry no direction                                                                                                                                                                                                                                                                               |

For Figure 3c the plotted quantity became `directionOfEffect * absEstimatedBeta`
— the same rescaled magnitude as `minorAlleleEstimatedBeta` without the
minor-allele flip.

#### The harmonisation claim, tested

The claim behind the approved R2-MJ-14 response — that all credible sets of a
variant report effects against the same allele — had never been tested. Four
checks over the 70,618 qualifying credible sets and their 40,706 lead variants:

| check                                                                                                   | result                                                                                            |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| lead variants with more than one `(ref, alt)` pair across their credible sets                           | **0** — the genotype coding is identical, as the variant id requires                              |
| same variant, same single disease, two or more studies: fully concordant pairs                          | **7,508 of 7,895 (95.1%)** on `originalBeta`, against 7,280 (92.2%) on `minorAlleleEstimatedBeta` |
| the same, restricted to the 262 pairs whose alternate-allele frequency straddles 0.5 across the studies | **237 (90.5%)** on `originalBeta`, against **9 (3.4%)** on `minorAlleleEstimatedBeta`             |
| the same, restricted to the 1,866 pairs whose studies have different major LD populations               | 1,719 (92.1%) against 1,491 (79.9%)                                                               |

The third row is the direct proof. Where the minor-allele flip is applied to
some of a variant's credible sets and not others, the harmonised beta still
agrees with itself nine times in ten on replicate associations to the same
disease, and the minor-allele referencing agrees one time in thirty. The
referencing, not the biology, produced the disagreement.

**How much of the data this reaches.** 3,239 of the 40,706 lead variants (8.0%)
have credible sets spanning more than one major LD population — 2,927 span two,
268 three, 34 four, 10 five. 452 (1.1%) have an alternate-allele frequency that
straddles 0.5 across their credible sets, which is exactly the set where the
flip is applied inconsistently; 334 of those are pleiotropic under the published
definition (3.4% of the 9,828). Recomputing `betaSignConcordance` with the flip
removed changes it for 292 of the 9,000 pleiotropic variants that have a value
under both, 241 of them downward. So the minor-allele referencing was not a
large error, but it was an error in one direction, and it is unnecessary.

#### Defined and excluded, and how far the representative moved

| quantity                                     | ungated `leadVPS` | amended `signedLeadVPS` |
| -------------------------------------------- | ----------------- | ----------------------- |
| lead variants with a lead_vPS                | 38,273            | **35,472**              |
| lead variants excluded (nothing contributes) | 2,433 (6.0%)      | **5,234 (12.9%)**       |
| lead variants with lead_vPS > 1              | 6,383             | **5,919**               |
| cluster representatives with a lead_vPS      | 18,791            | **17,022**              |
| cluster representatives excluded             | 1,250 (6.2%)      | **3,019 (15.1%)**       |
| cluster representatives with lead_vPS > 1    | 2,341             | **2,166**               |

For reference, `uniqueDiseases > 1` holds for 9,828 of the 40,706. The seed
representative differs from the most-diseases representative in **3,292 of
20,041 clusters (16.4%)** — 58.8% of the 5,595 that have more than one lead
variant. `leadVPS` is below `uniqueDiseases` for 1,845 variants and never above
it; `signedLeadVPS` is below `leadVPS` for 675 and never above it.

The score distribution over all 40,706 lead variants, since the gate recomputes
it rather than adjusting it:

| lead_vPS | published `uniqueDiseases` | ungated `leadVPS`  | amended `signedLeadVPS` |
| -------- | -------------------------- | ------------------ | ----------------------- |
| 1        | 30,878                     | 31,890             | 29,553                  |
| 2        | 6,114                      | 4,166              | 3,854                   |
| 3        | 1,988                      | 1,158              | 1,079                   |
| 4        | 696                        | 440                | 423                     |
| 5        | 341                        | 229                | 211                     |
| 6        | 193                        | 111                | 92                      |
| 7        | 109                        | 77                 | 68                      |
| 8        | 140                        | 47                 | 45                      |
| 9        | 50                         | 32                 | 29                      |
| 10       | 35                         | 21                 | 22                      |
| 11+      | 162                        | 102                | 96                      |
| —        | mean 1.480, max 85         | mean 1.320, max 71 | mean 1.320, max 71      |

The 10 row rises from 21 to 22 under the gate, which is the trimming effect made
visible: a variant loses a disease and lands on 10 from above.

**Retention at the top, for the response letter.** Of the 197 lead variants the
published definition calls highly pleiotropic (`uniqueDiseases >= 10`):

|                        | ungated `leadVPS` | amended `signedLeadVPS` |
| ---------------------- | ----------------- | ----------------------- |
| still ≥ 10             | 123               | **118**                 |
| still > 1              | 182               | **182**                 |
| fallen to 0 (excluded) | 11                | **11**                  |
| median retention       | 83.3%             | **80.0%**               |
| mean retention         | 74.1%             | **70.9%**               |

#### What did not move

Asserted in the notebooks rather than checked by eye, and re-verified after the
amendment: the 20,041 clusters and their membership, the 5,595 with more than
one lead variant, every disease-count statistic (R4.01-R4.05, range 1-120, mean
2.1415), every therapeutic-area statistic and the Spearman $\rho$ (R4.06-R4.09,
4,766 / 24% / max 19 / mean 1.4477), the cluster sizes and the seed lead variant
against `variant_clusters`, **R4.10-R4.13 and all twelve Figure 3b
coefficients** (asserted against literals in the notebook, so the rebuilt Figure
3 stands and was not rebuilt again — the gate cannot reach a model whose carrier
is chosen by smallest P value and whose outcome is the cluster's vPS over every
disease term; `plot_a.csv`, `plot_b.csv` and `cluster_covariates` are all
value-identical), the 46 pre-existing `variant_features` columns and the 11 of
the ungated family, the Supplementary Results 6 effect matrix (40,706 variants,
1,403 diseases before deduplication, 1,308 in the matrix, largest-$|\beta|$ per
disease-variant pair) and its modelling paragraph (S6.03-S6.06: 9,828, 24%, mean
1.48, max 85 — counted over all disease terms and deliberately ungated),
S6.09-S6.17, every gPS number with `results/gene_pleiotropy.json` untouched, and
every sheet in `06-supplementary-tables` except ST2.

**Registry tallies are unchanged at 608 PASS / 49 MISMATCH / 9 BLOCKED / 7
PRECOMPUTED.** The amendment moves values inside ids that were already MISMATCH
by design; no status flipped in either direction, so it introduces no new
by-design mismatch.

#### What moved

Figure 3a and 3b, and R4.10-R4.13, moved with the **representative** change and
are untouched by the sign gate. The "after" column below is final.

| id    | quantity                                              | published | before this change | after      |
| ----- | ----------------------------------------------------- | --------- | ------------------ | ---------- |
| R4.10 | vPS variance explained, predicted power alone (%)     | 14.7      | 14.669             | **15.073** |
| R4.11 | vPS variance explained, full joint model (%)          | 17.7      | 17.666             | **16.946** |
| R4.12 | vPS variance explained, power excluded (%)            | 6.0       | 5.962              | **3.827**  |
| R4.13 | vPS variance explained, max effective sample size (%) | 0.44      | 0.441              | **0.518**  |

Figure 3b, all twelve coefficients:

| covariate       | univariate before | univariate after | joint before | joint after |
| --------------- | ----------------- | ---------------- | ------------ | ----------- |
| Absolute beta   | 1.877             | **1.820**        | 0.331        | **0.382**   |
| MAF             | 0.398             | **0.375**        | 0.356        | **0.309**   |
| Sample size     | 0.806             | **0.776**        | 0.266        | **0.307**   |
| GERP            | 0.401             | **0.226**        | 0.147        | **0.028**   |
| PAV             | 0.750             | **0.601**        | 0.258        | **0.214**   |
| Predicted power | 1.392             | **1.442**        | 1.299        | **1.364**   |

GERP is the one qualitative change: in the joint model it goes from
$P = 2.9 \times 10^{-8}$ to $P = 0.29$, so it stops being a significant
independent predictor of vPS. Every other covariate keeps its sign, its rank
order and its significance. The most-diseases rule favoured the lead variant
with the most diseases in the cluster, and constrained variants carry more
diseases, so that rule was selecting GERP-high carriers.

The directionality block. Under the amendment there are **two** groups, not
three, because a variant either has a signed contributing credible set — in
which case its concordance is computable — or is excluded outright:

| id     | group                                      | published     | ungated `leadVPS` | amended `signedLeadVPS`         |
| ------ | ------------------------------------------ | ------------- | ----------------- | ------------------------------- |
| R4.14  | pleiotropic representatives (the universe) | 5,188         | 2,341             | **2,166**                       |
| R4.15  | fully concordant                           | 4,797 (92.5%) | 1,774             | **1,844 (85.1%)**               |
| R4.16  | discordant                                 | 391 (7.5%)    | 307               | **322 (14.9%)**                 |
| —      | concordance undefined                      | —             | 260               | **0 — no third group survives** |
| R4.14x | excluded, nothing contributes              | —             | 1,250             | **3,019**                       |
| R4.17  | representatives with lead_vPS $\geq 10$    | 135           | 68                | **67**                          |
| R4.18  | of those, agreement $\leq 0.8$             | 31            | 20                | **18**                          |
| R4.19  | genes carrying those variants              | 34            | 23                | **21**                          |

1,844 + 322 = 2,166 exactly, and the notebook asserts that every representative
with a `signedLeadVPS` also has a concordance, so the collapse to two groups is
verified rather than assumed. Note the concordant count **rises** (1,774 →
1,844) while the universe falls: variants that previously had no computable
concordance now have one, and most of them are concordant.

**Three effects are stacked here and can be separated.** Under the published
definition but on the new seed representative the block reads 3,983 pleiotropic
(3,457 with a concordance value, 526 without) / 3,002 concordant / 455
discordant / 91 at $\geq 10$ diseases / 23 below 0.8 / 25 genes. So the
representative change takes 5,188 to 3,983, the first redefinition takes it to
2,341, and the sign gate to 2,166.

**And the published 5,188 is now identified.** It is the number of cluster
representatives with `uniqueDiseases > 1` under the most-diseases rule
**without** requiring the variant to have a concordance value: 5,188 exactly, of
which 4,568 have one and 620 do not. That is why R4.14 never reproduced — the
notebook applied a `betaSignConcordance.notna()` filter the published count did
not. The published 4,797 / 391 split of those 5,188 still reconciles with
nothing: the same universe gives 3,952 concordant and 616 discordant, and adding
the 620 nulls to either side reaches neither 4,797 nor 391.

APOE:

| id    | quantity                            | published | ungated `leadVPS`                     | amended `signedLeadVPS`            |
| ----- | ----------------------------------- | --------- | ------------------------------------- | ---------------------------------- |
| R4.20 | 19_44908684_T_C lead_vPS            | 85        | 71                                    | **71**                             |
| R4.21 | 19_44908684_T_C $\beta$ concordance | 0.66      | 0.5714 (40 up / 30 down of 70 signed) | **0.5634 (40 up / 31 down of 71)** |
| R4.22 | 19_44908684_T_C therapeutic areas   | 15        | 15                                    | **15**                             |

This variant is the clearest case of the gate _changing_ rather than
subtracting. Its lead_vPS stays at 71, but one of those 71 diseases had an
unsigned credible set as its most significant study, so it counted toward
lead_vPS while contributing no direction. Gating that credible set out promotes
the disease's next-best study, which is signed and negative — hence 31 down
instead of 30, and a concordance of 40/71 rather than 40/70. It is **still the
most pleiotropic lead variant** by a wide margin: 71 against 47 for the
runner-up (`10_112998590_C_T`). The therapeutic-area count is unchanged at 15
and the **nine-negative / six-positive split still holds** exactly, area for
area.

**Figure 3 was not rebuilt again, and does not need to be.** `plot_a.csv`,
`plot_b.csv` and `cluster_covariates` are value-identical to the pre-amendment
run, and the twelve panel-b coefficients and R4.10-R4.13 are asserted in the
notebook. Panel c is also unaffected: the APOE export already drops credible
sets whose `originalBeta` is null, and the one gated-out APOE credible set is
exactly such a row, so `contributing` and the new `signedContributing` flag
agree on all 193 and 59 exported rows. The pixel comparison from the
pre-amendment rebuild therefore still stands: **1.01%** of pixels differ from
the published `figures/figure_3.pdf` (13,839 of 1,365,525 at 3x), against 0.03%
for the unchanged pipeline, with panels a and b accounting for almost all of it.

#### Sentences to change

**`sections/results/04_variant_pleiotropy.tex`, the variance-explained
sentence** — "predicted statistical power alone explained 14.7\% of vPS
variance, compared with 17.7\% for the full joint model and only 6.0\% when
excluded. Maximum effective sample size acted as a technical confounder with a
significant but relatively minor contribution to vPS ($R^2 = 0.44\%$)"

> **15.1\%**, **16.9\%** and **3.8\%**; $R^2 = 0.52\%$. The argument is
> unaffected and slightly strengthened: power alone now explains nearly all of
> what the joint model explains.

**The Figure 3b sentence** — "Univariate and joint linear models revealed
several functional factors positively associated with vPS: GERP constraint, PAV
effects, MAF, maximum effective sample size, the maximum absolute effect size
($|\beta|$), and predicted statistical power"

> GERP has to move or be qualified: it is still positive and significant
> univariately ($\beta = 0.23$, $P = 8 \times 10^{-16}$) but is no longer
> significant in the joint model ($\beta = 0.03$, $P = 0.29$).

**The directionality sentence** — "Of the 5,188 pleiotropic lead variants, 4,797
(92.5\%) showed fully concordant directionality across their associated diseases
\dots the remaining 391 (7.5\%) displayed at least one opposing direction"

> **Of the 2,166 pleiotropic lead variants, 1,844 (85.1\%) showed fully
> concordant directionality and 322 (14.9\%) displayed at least one opposing
> direction.** There is no third group: every counted disease carries a signed
> effect, so the two numbers exhaust the universe and the denominator needs no
> qualification. A further **3,019 cluster representatives have no credible set
> from a single-disease study with a signed effect** and are outside the
> analysis entirely; that count belongs in the sentence or its Methods. The
> published 92.5\% counted variants that were concordant by construction and, on
> the previous representative rule, is not recoverable at all.

**The high-pleiotropy sentence** — "Of 135 lead variants with lead_vPS
$\geq 10$, 31 variants (23\%) associated with 34 genes showed directionality
agreements of less than 80\%"

> **Of 67 lead variants with lead_vPS $\geq 10$, 18 (27\%) associated with 21
> genes.** The "less than 80\%" wording needs to become "80\% or less": the code
> selects on $\leq 0.8$, and one of the 18 sits exactly at 0.8 (`RGL3`,
> `19_11416089_T_G`), so the operator is load-bearing — `< 0.8` would give 17.
> Supplementary Table 2 was reissued on this same universe on 2026-08-22 and now
> matches these three numbers row for row; see
> `chapters/06-supplementary-tables/README.md`.

**The APOE sentence** — "Variant 19_44908684_T_C in \textit{APOE}
$\varepsilon$4 was the most
pleiotropic lead variant (lead\_vPS $= 85$; $\beta$
concordance $= 0.66$), showing associations with 15 therapeutic areas in both
directions: nine with predominantly negative effects (decreased risk) and six
with positive effects (increased risk)"

> **lead_vPS $= 71$; $\beta$ concordance $= 0.56$.** "15 therapeutic areas" and
> the nine/six split are unchanged. It is still the most pleiotropic lead
> variant, 71 against 47 for the runner-up.

**Methods, and Supplementary Methods where lead_vPS is defined** — the
definition itself has to be restated: a credible set contributes only if the
variant is its lead variant, its study maps to exactly one disease term, and its
direction of effect is known; direction comes from the harmonised effect-allele
beta with the most significant contributing study winning per disease;
concordance is the largest same-direction share and is defined wherever lead_vPS
is at least 1; and variants with nothing contributing are excluded and counted,
not scored 0.

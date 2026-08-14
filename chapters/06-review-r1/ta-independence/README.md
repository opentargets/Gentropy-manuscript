# Are diseases in different therapeutic areas more independent than diseases in the same one?

This is the assumption behind gps_TA, and round-1 referee 2 disputes it
directly:

> **R2-MJ-12** — "Diseases linked through a single cluster are likely to be
> pathologically correlated; I am not convinced that spread across therapeutic
> areas captures real diversity."

> **R2-MJ-3** — "Also, diseases in different therapeutic areas may share
> aetiology, for example fibrosis or inflammation, so spread across therapeutic
> areas is not evidence that horizontal pleiotropy is reduced."

The genetic-correlation matrix S settles it without any modelling: take its
disease traits, map each to one therapeutic area, and compare the genetic
correlation of **within-area** disease pairs against **between-area** pairs.

## Re-run 2026-08-14 on the corrected matrix

S was rebuilt the same day this notebook was first re-executed against it:
`canonical_pairwise_table.parquet` carried stale disease ids from an earlier
Open Targets release, so 222 of its 498 disease-labelled traits had no row in
`disease.parquet` at all. The fix (join `studyId1`/`studyId2` to the 25.06 study
index instead of trusting the pair table's own `diseaseId_1`/`diseaseId_2`) took
S from 1,066 to **1,114** traits — see
`chapters/02-analysis/08-genecorrs/01-gene-corrs-preparation.ipynb` and the
effective-independent-traits folder's README for the fuller account.

**Net effect here: coverage more than doubled (156/498 → 400/551 diseases
mapped), the effect size weakened slightly but stayed solidly significant, and
the effective-number statistic (statistic 4) flipped from non-significant to
significant.** Unlike the effective-independent-traits analysis, nothing here
reverses — every headline conclusion still points the same way, some more
strongly than before.

## Headline — the assumption holds, but only partly

**Within-area disease pairs are substantially more genetically correlated than
between-area pairs, and crossing a therapeutic-area boundary cuts mean
|r<sub>g</sub>| by 21%** (was 23%). On **400** diseases, **5,844** within-area
and **73,956** between-area pairs (was 156 diseases, 912/11,178 pairs):

| Statistic                          | Within area | Between area |                                                                           |
| ---------------------------------- | ----------- | ------------ | ------------------------------------------------------------------------- |
| mean \|r<sub>g</sub>\|             | **0.401**   | **0.317**    | difference **+0.084**, permutation **P = 0.0001** (floor at 10,000 draws) |
| median \|r<sub>g</sub>\|           | 0.296       | 0.205        |                                                                           |
| share with \|r<sub>g</sub>\| ≥ 0.2 | 61.0%       | 50.7%        | **1.20×**                                                                 |
| share with \|r<sub>g</sub>\| ≥ 0.5 | **33.1%**   | **22.4%**    | **1.48×**                                                                 |
| probability of superiority         | —           | —            | **0.574**                                                                 |

_Probability of superiority_: pick one within-area pair and one between-area
pair at random — the within-area pair is the more correlated one **57% of the
time** (was 60%; 0.5 would mean no difference).

Every number moved a little towards the null (mean difference 0.105 → 0.084,
superiority 0.597 → 0.574, ≥0.5 fold 1.55× → 1.48×) — expected, since the
newly-recovered 244 diseases are a less curated population than the original 156
survivors, and are diluting the effect somewhat. **The permutation P is
unchanged in substance: still the floor of the test (0 of 10,000 permutations
reached the observed difference).**

**The honest qualification, unchanged:** between-area pairs are _not_
independent either — mean |r<sub>g</sub>| 0.317, and over a fifth still exceed
0.5. Therapeutic-area spread remains a genuine but **partial** proxy for genetic
independence.

**This is still not just disease subtypes.** Dropping the (now much larger) set
of ontology parent/child pairs — **1,007** pairs (741 within-area, 266
between-area), versus 68 before — leaves the result intact: mean difference
**0.0735** (0.390 vs 0.316), superiority **0.563**, still well above null.

## Why coverage jumped from 31% to 73% — and it is not a new mapping trick

**400 of S's 551 disease traits (72.6%) now carry a real therapeutic area**,
versus 156 of 498 (31%) before. The entire gain is explained by the stale-id
fix, not by any change to the area-mapping rule:

| Reason                                    | Before (1,066-trait S)          | After (1,114-trait S)                  |
| ----------------------------------------- | ------------------------------- | -------------------------------------- |
| absent from the disease ontology entirely | 222 (210 MONDO, 9 OBA, 3 EFO)   | **0**                                  |
| in the ontology, but under no area root   | 120 (76 HP, 37 EFO, 6 GO, 1 MP) | 151 (74 HP, 68 EFO, 6 GO, 1 MP, 2 OBA) |
| mapped to a real area                     | 156                             | **400** (95 MONDO, 291 EFO, 14 HP)     |

**Every MONDO term in S now maps (95 of 95, 0 absent, 0 present-but-unmapped)**
— the stale-id problem that made 210 MONDO terms look ontology-absent is gone
entirely, because those terms were never actually missing from the ontology;
they were the wrong (superseded) ids. What remains as "other" is now exclusively
the **second** cause from before: terms genuinely in the ontology but sitting
outside any disease-area hierarchy (mostly HP phenotype codes, which sit under
`phenotype` by construction, plus some EFO measurement-adjacent terms).

The PheCode / non-disease composition also shifted: **155 of 551 (28%) are
PheCode-derived** (91 of them map) — a smaller share than before (266 of 498,
53%), because the newly-recovered traits are disproportionately "real" disease
terms rather than PheCode phenotypes. **35 of 551 are UK Biobank questionnaire /
family-history fields** (down from 52 of 498) — e.g. _"Have you ever been
pregnant?"_, _"Usual side of head for mobile phone use: Right"_, _"Invitation to
physical activity study, acceptance"_ — still not diseases, still miscategorised
by the upstream `measurement` flag.

The upstream `therapeutic_area` column (in `canonical_pairwise_table.parquet`,
unaffected by the id fix since it's computed at the study level before ids are
assigned) independently gives **114 of 551 (20.7%)** `other` against this
notebook's **151 of 551 (27.4%)** — a gap of 37 terms, wider than the 5-term gap
before. Not investigated further here; both numbers describe roughly a quarter
to a fifth of S's disease traits as unclassified, and the direction of the gap
(this notebook's hierarchy is stricter) is the same as before.

**Root cause of the composition is unchanged**: S keeps only NFE-ancestry,
heritability-filtered studies with `n_snps_used ≥ 100,000`, selecting large
well-powered biobank PheWAS scans.

## Distribution of the 400 diseases across areas (`ta_independence_distribution-r1.csv`)

**21 of the 22** non-measurement areas are represented (was 20), **all 21** with
at least 2 diseases (was 19):

| Therapeutic area                             | Diseases | Within-area pairs |
| -------------------------------------------- | -------- | ----------------- |
| cancer or benign tumor                       | 59       | 1,711             |
| cardiovascular disease                       | 45       | 990               |
| gastrointestinal disease                     | 40       | 780               |
| musculoskeletal or connective tissue disease | 30       | 435               |
| nervous system disease                       | 29       | 406               |
| infectious disease                           | 23       | 253               |
| immune system disease                        | 22       | 231               |
| sign or symptom                              | 22       | 231               |
| disorder of visual system                    | 20       | 190               |
| injury, poisoning or other complication      | 17       | 136               |
| nutritional or metabolic disease             | 16       | 120               |
| respiratory or thoracic disease              | 14       | 91                |
| reproductive system or breast disease        | 13       | 78                |
| integumentary system disease                 | 13       | 78                |
| urinary system disease                       | 12       | 66                |
| endocrine system disease                     | 7        | 21                |
| pancreas disease                             | 5        | 10                |
| hematologic disease                          | 5        | 10                |
| pregnancy or perinatal disease               | 3        | 3                 |
| psychiatric disorder                         | 3        | 3                 |
| disorder of ear                              | 2        | 1                 |

Only **one** area is now absent from S's disease traits entirely: genetic,
familial or congenital disease (pregnancy or perinatal disease, previously also
absent, now has 3 diseases).

## Method

Unchanged from before:

- **Therapeutic-area map** — one area per disease, `therapy_area_hierarchy`
  priority order from
  `chapters/01-data-preparation/04_qualifying_dataset_generation.ipynb` (first
  match wins), resolved from `disease.parquet` descendants. Identical rule to
  `../disease-subsampling/`.
- **|r<sub>g</sub>| not r<sub>g</sub>** — a strong negative correlation is just
  as much a failure of independence.
- **Permutation P value** — the area label is shuffled over _diseases_, not over
  pairs, 10,000 draws. Disease pairs share diseases and are not independent
  observations, so a t-test would be invalid.
- **Zero-filled cells** — now 41 of 79,800 pairs (15 within, 26 between) have no
  measured r<sub>g</sub> and sit at 0 (was 9 of 12,090). Dropping them changes
  nothing (third row of `ta_independence_comparisons-r1.csv`: mean difference
  0.0852 vs 0.0843 including them).

## Statistic 4 — effective-number efficiency: now significant, not just directional

The same question in the currency the manuscript already uses. For each area
with k ≥ 2 diseases, the Li & Ji Meff of that area's submatrix divided by k,
against 1,000 size-matched random disease sets drawn from the whole pool.

|                            | Within area           | Size-matched null     |
| -------------------------- | --------------------- | --------------------- |
| mean efficiency (Meff / k) | **0.899** (was 0.836) | **0.971** (was 0.937) |
| k-weighted efficiency      | **0.922** (was 0.850) | **0.998** (was 0.946) |

- **14 of 21** areas fall below their matched null (was 11 of 19); **6**
  individually at P < 0.05 (was 5): cardiovascular disease (P = 0.0020), sign or
  symptom (P = 0.0010), respiratory or thoracic (P = 0.0020), nutritional or
  metabolic disease (P = 0.0040), endocrine system disease (P = 0.0020),
  gastrointestinal disease (P = 0.0140)
- **paired Wilcoxon across the 21 areas: P = 0.0113 — now significant** (was P =
  0.064, not significant). This is the one genuine strengthening from the
  rebuild: statistic 4 is no longer "consistent supporting detail" only, it is
  now an independent confirmation in its own right.
- two areas that were individually significant before are not now: infectious
  disease (0.924 vs 0.991, P = 0.149, was P = 0.005) and nervous system disease
  (0.920 vs 1.004, P = 0.081, was P = 0.020) — with more diseases per area the
  within-area sets are less unusual for those two specific areas.
- **counterexample still holds: cancer or benign tumor**, now the largest area
  at k = 59 (was 23), still sits _above_ its null (1.082 vs 1.048, was 1.036 vs
  0.982). The cancers in S are still not notably correlated with one another,
  and this is now based on a much larger, more stable sample.

So statistic 4 now agrees with statistics 1–3 **and clears its own significance
threshold** — a stronger position than before, when it was explicitly flagged as
directional-only.

## What to say to the referee

1. **The direction is established and the effect is not small, on a much larger
   and more representative disease sample than before.** Same-area disease pairs
   average |r<sub>g</sub>| 0.401 against 0.317 across areas, P = 0.0001 by
   permutation (the test's floor), and 1.48× as many same-area pairs exceed
   |r<sub>g</sub>| 0.5. Therapeutic-area spread does track genetic independence.
2. **It is not an artefact of disease subtypes.** Removing 1,007 ontology
   parent/child pairs leaves the result intact.
3. **Concede the residual, slightly larger than before.** Between-area pairs
   still average 0.317, so cross-area spread is a partial proxy — if anything a
   little more so with the fuller disease list.
4. **The effective-number version is now a genuine second line of evidence, not
   just consistent with the first.** Paired Wilcoxon P = 0.0113 across 21 areas;
   cancer remains a disclosed counterexample.

## Limitations

- **27% of S's disease traits are still unusable here** (`other`, was 69%) —
  down substantially, and now entirely explained by "in the ontology but under
  no area root" (mostly HP phenotype codes), not by any ontology-coverage gap.
- **21 areas is still a small sample** for the area-level test, though larger
  than the 19 before, and area sizes remain very uneven (59 down to 2).
- `sign or symptom` and `injury, poisoning or other complication` are not
  therapeutic areas in a clinical sense but are retained because the published
  hierarchy treats them as such; `sign or symptom` remains one of the strongest
  contributors to the statistic-4 result.
- Single-area-per-disease loses information: a multi-membership version was not
  run (unchanged limitation).
- S itself covers only NFE-ancestry, heritability-filtered studies, so this is a
  statement about that subset of GWAS (unchanged limitation).
- **The magnitude of the headline effect is now known to be sensitive to
  matrix/id corrections** — it weakened somewhat (mean difference 0.105 → 0.084)
  as coverage improved. Any further ontology-duplicate cleanup
  (`[[project_ontology_duplicates]]`) should be expected to move these numbers
  again, though the direction has been stable across two independent matrix
  builds now.

## Notebook

| Notebook                        | Needs Spark | Runtime | What it does                                |
| ------------------------------- | ----------- | ------- | ------------------------------------------- |
| `01_within_vs_between_ta.ipynb` | no          | ~1 min  | TA map, pair classification, statistics 1–4 |

Imports `meff_li_ji` from `../effective-independent-traits/eit_lib.py`, so the
estimator is the one already validated there (including its documented floor
near 2 for duplicate clusters).

```bash
cd chapters/06-review-r1/ta-independence
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 01_within_vs_between_ta.ipynb
```

## Exports (all in `data/intermediate_files/`)

| File                                        | Contents                                                                                          |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `ta_independence_distribution-r1.csv`       | diseases per therapeutic area, and within-area pair counts                                        |
| `ta_independence_other_breakdown-r1.csv`    | why each of the 551 disease traits does or does not get an area, by ontology prefix               |
| `ta_independence_comparisons-r1.csv`        | within vs between \|r<sub>g</sub>\|, superiority, tail fractions, permutation P — three pair sets |
| `ta_independence_efficiency-r1.csv`         | Meff / k per area against the size-matched cross-area null                                        |
| `ta_independence_efficiency_summary-r1.csv` | areas below the null, weighted averages, paired Wilcoxon                                          |

All exports regenerated 2026-08-14 against the rebuilt (1,114-trait) matrix.

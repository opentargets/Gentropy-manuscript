# Supplementary tables

Four notebooks, and three places a sheet can live.

| Notebook                        | Builds                                                                      |
| ------------------------------- | --------------------------------------------------------------------------- |
| `01_supplementary_tables.ipynb` | the sheets that come straight from the pipeline, into `sheets/`             |
| `02_manual_table_numbers.ipynb` | recomputes the counts inside the hand-made sheets, into `manual/refreshed/` |
| `03_l2g_tables.ipynb`           | ST12 and ST13, which need the L2G training set and held-out split           |
| `04_fine_mapping_numbers.ipynb` | the cells of ST10, against the published values transcribed in the notebook |

- `sheets/` — built by the pipeline. Overwritten on every run.
- `manual/` — sheets assembled by hand, kept verbatim; the notebooks write
  comparisons beside them and never edit them. See `manual/README.md`.
- `assets/` — static spreadsheets shipped as submitted (ST3, ST11); not rebuilt,
  not recomputed.

```bash
tools/run_chapter.sh chapters/06-supplementary-tables
```

## Verified against the published workbook

Compared cell by cell against
`~/Projects/manuscript_gentropy/20260814_1238_supplementary_tables.xlsx` on
2026-08-20. That workbook carries a title row, a blank, a column data-dictionary
block, a blank, then the header — so a sheet's data row count is its `max_row`
minus the dictionary block.

Only ST1, ST7, ST8, ST10 and ST12 had a comparison inside a notebook before
this; the rest were checked on row counts alone, which is why this table exists.

| Sheet                                            | Verdict                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ST1 studies                                      | **exact** — 100,526 x 12, keys aligned. The only differences are formatting: dates stored as datetimes in the workbook, list columns serialised as `['a' 'b']` here against `['a', 'b']` there, and 400 `traitFromSource` cells where the **published sheet** is mojibake (`Alzheimer‚Äôs`) and this one is correct                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ST2 discordant variants                          | **37 rows against the published 31.** Reissued on the all-lead-variant universe with the amended `signedLeadVPS` / `signedLeadDirectionalConcordance` (see below), over 29 colocalisation clusters and 37 distinct genes. `isClusterRepresentative` flags the 18 rows the main-text sentence is computed over, so both universes read off the one sheet. Earlier builds: 18 rows on the representative universe, 39 on the first redefinition, 59 published                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ST3 GSEA                                         | out of scope — hand-written, static asset                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ST4 gPS gene categories                          | **exact** — 21 rows, 0 differing cells across the 9 shared columns. The published sheet carries a `count` column this one does not, and this one adds `fdr`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ST5 ChEMBL target-indication pairs               | **exact** — 37,377 x 11, **0 differing cells**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ST6 drug-target enrichment                       | **odds ratios exact** — all 33 published rows match to six decimals. Three differences: `total_indirect_assoc` holds a different quantity under the same name (742 here against 151,704 there, i.e. the supported subset rather than the full universe), one `p_value` differs tenfold, and this sheet adds the two therapeutic-area rows (`TA-1_subEvid`, `TA-6plus_subEvid`) the published sheet omits                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ST7 PAV gene-disease pairs                       | **exact** — 4,742 x 17, 0 differing cells once both are sorted on the full key. Unsorted comparison shows ~1,100 rows differing because row order within a tied `(variantId, diseaseId)` group is arbitrary in both. `diseaseIds` is serialised `a;b` here and `['a', 'b']` there                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ST8 subgroup analysis                            | **965 against the published 1,001** on subtable 3. See `manual/README.md`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ST9 therapeutic-area assignment                  | **exact** — 24 rows; one cell where the published sheet writes `N/A` and this one leaves the `other` root blank                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ST10 fine-mapping statistics                     | **54 of 54** checkable cells, by `04_fine_mapping_numbers.ipynb`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ST11 colocalisation overlap                      | out of scope — hand-compiled, static asset, definitions unrecovered                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ST12 L2G performance                             | **8 of 8** rows exact, by `03_l2g_tables.ipynb`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ST15 `cluster_id` renumbering (2026-08-22)       | **Reverted the same day, no change stands.** The representative rank key was briefly moved to `chi2Stat`, which permuted 22 of the 20,041 cluster ids and moved this sheet's `cluster_id` column. The revert restores the sheet byte-for-byte; `diff -rq` against the pre-change sheets is clean. See `chapters/02-analysis-main/README.md`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ST13, ST14, ST15                                 | **not in the published workbook** — the manuscript marks them `TODO(data)`, so there is nothing to compare against. Built here from the pipeline: 12,274 / 36,858 / 42,918 rows                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ST16 trait distribution across therapeutic areas | **was exact in content; 18 of the 22 area rows now differ.** It was verified exact on 2026-08-20, when the counts still came from the legacy therapeutic-area assignment. `clusters.therapeutic_area_lookup()` now defaults to the Supplementary Table 9 order (see `chapters/02-analysis-main/README.md`), and ST16 reads that lookup, so `immune system disease` goes 111/75 to 29/15 and `genetic, familial or congenital disease` 8/2 to 180/121. The totals hold at 2,320 and 1,394. The published sheet's own column description says the assignment follows "the hierarchy in Supplementary Table 9", which the sheet did not do and this one now does — so this is a correction, not a regression, but **it changes a submitted sheet and is the one open decision**: pin the ST16 cell back to `primaryTherapeuticAreaLegacy` to restore the published counts, or reissue the sheet. Row order was already on the Supplementary Table 9 order, unlike the published sheet |

So **seven sheets reproduce exactly** (ST1, ST4, ST5, ST7, ST9, ST10, ST12 — the
last one modulo an `N/A` placeholder), one is exact on its statistics but not
its columns (ST6), three disagree (ST2, ST8, and ST16 since the therapeutic-area
order was unified), three have no published counterpart (ST13-15) and two are
out of scope (ST3, ST11).

`ST6`'s `total_indirect_assoc` is worth fixing before resubmission: it is
cosmetic and it makes the sheet look wrong to a reader comparing it with the
published version. `ST16` is not cosmetic — its counts moved with the
therapeutic-area order, and the row above says what has to be decided.

`ST14` also carries a `therapeuticArea` label from the same lookup, so 5,470 of
its 36,858 rows (14.8%, over 164 diseases) changed label with it. ST14 has no
published counterpart, and its `geneId`, `diseaseId`, `gPS` and
`numberOfTherapeuticAreas` columns are untouched.

## Manuscript text that needs updating

### ST2 — reissued on the all-lead-variant universe (2026-08-22, superseding the representative build)

The sheet selects **all lead variants** with `signedLeadVPS >= 10` and
`signedLeadDirectionalConcordance <= 0.8`: **37 rows over 29 colocalisation
clusters, 37 distinct L2G-prioritised genes**. All three are asserted in the
notebook. The representative rule is unchanged (stored P then lead-variant PIP)
and nothing else about the analysis moved.

This reverses the 2026-08-22 morning decision to build on the 20,041 cluster
representatives. That build is not lost: `isClusterRepresentative` marks its 18
rows, and the flagged subset was verified **identical to the 18-row sheet on all
18 shared columns** with the same 18 `variantId`s, so the main-text universe is
still readable off the sheet. The notebook asserts the flag reproduces the
representative selection and that 67 representatives clear `lead_vPS >= 10`
(R4.17).

| universe                                                       | rows   | clusters | genes  |
| -------------------------------------------------------------- | ------ | -------- | ------ |
| all lead variants — **the sheet**                              | **37** | **29**   | **37** |
| cluster representatives — the main-text sentence (R4.17-R4.19) | 18     | 18       | 21     |
| published sheet                                                | 31     | —        | —      |

Filter steps, over all 40,706 lead variants: 35,472 have a lead_vPS at all, 118
reach `lead_vPS >= 10`, 37 of those are at concordance `<= 0.8`, and 18 of the
37 are representatives. `<= 0.8` is deliberate and matches the main text;
**three rows sit exactly at 0.8** here (TERT `5_1285859_C_A`, FUT2
`19_48703417_G_A`, RGL3 `19_11416089_T_G`), against one in the 18-row build, so
`< 0.8` would give 34 rather than 37.

**`cluster` is carried so duplicated loci are visible.** Six clusters contribute
more than one row, and in three of them the representative does not qualify at
all:

| cluster | rows                                                    | representative     | representative qualifies |
| ------- | ------------------------------------------------------- | ------------------ | ------------------------ |
| 104     | 3 (HECTD4, PTPN11, ALDH2/TRAFD1)                        | `12_112379979_T_A` | yes                      |
| 121     | 3 (TERT ×3)                                             | `5_1292868_C_A`    | **no**                   |
| 26      | 2 (APOE ×2, including `19_44908684_T_C` at lead_vPS 71) | `19_44888997_C_T`  | **no**                   |
| 103     | 2 (ABO ×2)                                              | `9_133274295_A_T`  | **no**                   |
| 307     | 2 (FUT2 ×2)                                             | `19_48702915_C_T`  | yes                      |
| 350     | 2 (SERPINA1, SERPINA1/SERPINA2)                         | `14_94371805_G_T`  | yes                      |

So this universe is what puts the APOE and ABO loci in the table — they were
absent from the 18-row build because their clusters are represented by variants
that fail the `lead_vPS >= 10` cut. Rows are ordered by `cluster`, then
concordance ascending, so each locus reads as a block.

**Which column each field reads.** Unchanged from the representative build:
every count and score is amended-family (`signedLeadVPS`,
`signedLeadUniqueTherapeuticAreas`, `signedLeadDirectionalConcordance`,
`signedLeadUpDiseases`, `signedLeadDownDiseases`, `signedLeadTherapeuticAreas`),
with `leadVPS`, `leadDirectionalConcordance`, `uniqueDiseases`,
`uniqueTherapeuticAreas` and `betaSignConcordance` retained for provenance and
used in no selection. Gene symbols come from `prioritisedGenes` mapped through
`gene_table.approvedSymbol`; `cluster` from `cluster_membership`, and the
representative of each cluster from `variant_clusters.leadVariantId`.

**The two gate checks were re-run on the 37 rows and both pass.** 617 gated
contributing associations sit behind them; the recomputed per-variant disease,
increased-risk and decreased-risk counts **agree with `signedLeadVPS`,
`signedLeadUpDiseases` and `signedLeadDownDiseases` for all 37**, and **no row
has an empty direction** (0 of 37) — so every row can illustrate discordance.
The example columns are additionally asserted to draw only from single-disease
studies with a non-null `directionOfEffect`:
`assert gated["diseaseIds"].map(len).eq(1).all()` and
`assert gated["direction"].notna().all()`.

Every other sheet is byte-identical — verified file by file against the
pre-change `sheets/`. Registry unchanged at 602 PASS / 55 MISMATCH / 9 BLOCKED /
7 PRECOMPUTED, the 55 including S7.04-S7.09 from the SR 7 adoption.
`tools/expected_numbers.tsv` untouched.

#### What the representative filter costs: the cited examples (read-only audit, 2026-08-22)

The paragraph citing ST2 names APOE, ABO, GCKR and TYK2. Only **GCKR
`2_27508073_T_C`** is in the 18 rows, and it is there because it _is_ its
cluster's representative (cluster 144, 15 lead variants). The other three are
absent for two different reasons, which the wording has to keep apart:

- **APOE and ABO fail only the representative filter.** Both clear the
  thresholds comfortably — APOE `19_44908684_T_C` at lead*vPS 71 / concordance
  0.5634, ABO `9_133257521_T_TC` at 20 / 0.7000 and `9_133274293_AC_A` at 12 /
  0.7500 — and in each case the cluster's representative is a \_different*
  variant that fails the lead_vPS ≥ 10 cut: `19_44888997_C_T` at lead_vPS 8
  (cluster 26, 26 lead variants) and `9_133274295_A_T` at lead_vPS 4 (cluster
  103, 51 lead variants).
- **TYK2 and the second APOE variant fail on their own numbers**, regardless of
  the universe. `19_10355447_C_T` has lead_vPS 2, and `19_44908822_C_T` has
  concordance 0.9565. Neither would appear under either universe.

**The APOE representative is decided by a PIP difference of 1.8 parts per
billion.** Three credible sets in cluster 26 tie at the smallest P value, and
all three report it as `1.0e-323` — the double-precision floor, so the true P
values are unresolvable and the P comparison carries no information between
them. The tie therefore falls entirely to the lead-variant PIP, where
`19_44888997_C_T` scores 1.0000000000 twice and `19_44908684_T_C` scores
0.9999999982. That single comparison is what keeps the most pleiotropic lead
variant in the dataset out of the table.

For reassurance on the rule generally: 88 of the 20,041 clusters have a seed
chosen from a residual tie on both keys, and in **none** of them does the tie
span more than one lead variant — so the representative _variant_ is never
ambiguous even where the seed credible set is.

The all-lead-variant universe would be 37 rows over **29 distinct clusters**:
the 18 representatives are a strict subset, and the 19 additions include 6
clusters contributing more than one row — TERT three times, HECTD4/PTPN11/ALDH2
three times in cluster 104, and APOE, ABO, SERPINA1 and FUT2 twice each. Full
tables in the session audit; the wording decision is the author's.

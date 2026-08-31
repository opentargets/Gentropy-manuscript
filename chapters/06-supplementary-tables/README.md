# Supplementary tables

Four notebooks, and three places a sheet can live.

| Notebook                        | Builds                                                                      |
| ------------------------------- | --------------------------------------------------------------------------- |
| `01_supplementary_tables.ipynb` | the sheets that come straight from the pipeline, into `sheets/`             |
| `02_manual_table_numbers.ipynb` | recomputes the counts inside the hand-made sheets, into `manual/refreshed/` |
| `03_l2g_tables.ipynb`           | ST12 and ST13, which need the L2G training set and held-out split           |
| `04_fine_mapping_numbers.ipynb` | the cells of ST10, against the published values transcribed in the notebook |

- `sheets/` — built by the pipeline. Overwritten on every run. One sheet is
  written from outside this chapter: **ST17** comes from
  `chapters/03-analysis-supplementary/14_genetic_correlation.ipynb`, which holds
  the genetic correlation matrix and the therapeutic-area assignment it needs.
  On 2026-08-24 it was rebuilt on the Supplementary Table 9 hierarchy along with
  the rest of Supplementary Results 14.2 — 253 cells over 22 areas, where the
  legacy hierarchy gave 231 over 21. See that chapter's README.
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

| Sheet                                            | Verdict                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ST1 studies                                      | **exact** — 100,526 x 12, keys aligned. The only differences are formatting: dates stored as datetimes in the workbook, list columns serialised as `['a' 'b']` here against `['a', 'b']` there, and 400 `traitFromSource` cells where the **published sheet** is mojibake (`Alzheimer‚Äôs`) and this one is correct                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ST2 discordant variants                          | **37 rows against the published 31**, and `Number of associated therapeutic areas` reissued on the Supplementary Table 9 hierarchy (19 rows up, total 231 -> 252; see below). Reissued on the all-lead-variant universe with the amended `signedLeadVPS` / `signedLeadDirectionalConcordance` (see below), over 29 colocalisation clusters and 37 distinct genes. `isClusterRepresentative` flags the 18 rows the main-text sentence is computed over, so both universes read off the one sheet. Earlier builds: 18 rows on the representative universe, 39 on the first redefinition, 59 published                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ST3 GSEA                                         | out of scope — hand-written, static asset                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ST4 gPS gene categories                          | **exact** — 21 rows, 0 differing cells across the 9 shared columns. The published sheet carries a `count` column this one does not, and this one adds `fdr`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ST5 ChEMBL target-indication pairs               | **exact** — 37,377 x 11, **0 differing cells**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ST6 drug-target enrichment                       | **odds ratios exact** — all 33 published rows match to six decimals, and `total_indirect_assoc` now agrees on all 33 as well (fixed 2026-08-25, see below). Two differences remain: on the twelve forest strata rows the published `p_value` is the z-test P, which this sheet carries in `z_p_value` while `p_value` holds Fisher's exact, and the same twelve rows' `ci_low` / `ci_high` differ in the fifth to sixth decimal (largest relative difference 1.2e-5, on `MoreBigEffect_subEvid`). This sheet also adds the two therapeutic-area rows (`TA-1_subEvid`, `TA-6plus_subEvid`) the published sheet omits                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ST7 PAV gene-disease pairs                       | **exact** — 4,742 x 17, 0 differing cells once both are sorted on the full key. Unsorted comparison shows ~1,100 rows differing because row order within a tied `(variantId, diseaseId)` group is arbitrary in both. `diseaseIds` is serialised `a;b` here and `['a', 'b']` there                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ST8 subgroup analysis                            | **965 against the published 1,001** on subtable 3. See `manual/README.md`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ST9 therapeutic-area assignment                  | **exact** — 24 rows; one cell where the published sheet writes `N/A` and this one leaves the `other` root blank                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ST10 fine-mapping statistics                     | **54 of 54** checkable cells, by `04_fine_mapping_numbers.ipynb`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ST11 colocalisation overlap                      | out of scope — hand-compiled, static asset, definitions unrecovered                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ST12 L2G performance                             | **8 of 8** rows exact, by `03_l2g_tables.ipynb`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ST15 `cluster_id` renumbering (2026-08-22)       | **Reverted the same day, no change stands.** The representative rank key was briefly moved to `chi2Stat`, which permuted 22 of the 20,041 cluster ids and moved this sheet's `cluster_id` column. The revert restores the sheet byte-for-byte; `diff -rq` against the pre-change sheets is clean. See `chapters/02-analysis-main/README.md`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ST13, ST14, ST15                                 | **not in the published workbook** — the manuscript marks them `TODO(data)`, so there is nothing to compare against. Built here from the pipeline: 12,274 / 36,858 / 42,918 rows                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ST16 trait distribution across therapeutic areas | **was exact in content; 18 of the 22 area rows now differ.** It was verified exact on 2026-08-20, when the counts still came from the legacy therapeutic-area assignment. `clusters.therapeutic_area_lookup()` now defaults to the Supplementary Table 9 order (see `chapters/02-analysis-main/README.md`), and ST16 reads that lookup, so `immune system disease` goes 111/75 to 29/15 and `genetic, familial or congenital disease` 8/2 to 180/121. The totals hold at 2,320 and 1,394. The published sheet's own column description says the assignment follows "the hierarchy in Supplementary Table 9", which the sheet did not do and this one now does — so this is a correction, not a regression, but **it changes a submitted sheet**: the choice was to pin the ST16 cell back to the legacy hierarchy column to restore the published counts, or reissue the sheet. Row order was already on the Supplementary Table 9 order, unlike the published sheet. **Resolved 2026-08-24 in favour of reissuing** — see "The ST16 hierarchy question is settled" below |

So **seven sheets reproduce exactly** (ST1, ST4, ST5, ST7, ST9, ST10, ST12 — the
last one modulo an `N/A` placeholder), one is exact on its statistics but not
its columns (ST6), three disagree (ST2, ST8, and ST16 since the therapeutic-area
order was unified), three have no published counterpart (ST13-15) and two are
out of scope (ST3, ST11).

`ST16` is not cosmetic — its counts moved with the therapeutic-area order, and
the row above says what has to be decided.

## `ST6`'s `total_indirect_assoc` holds the total again (2026-08-25)

The column names the size of the indirect-association universe an enrichment is
computed in, one constant per evidence source. That is what
`chemblDrugEnrichment.drug_enrichemnt_from_evidence` returns, and the eighteen
`resources` rows of this sheet carried it correctly all along — 60,318 for OMIM,
15,304 for the gene-based tests. The seventeen forest rows did not: Results 6
builds them itself from the pair-level table, has no such column, and renamed
its per-stratum count of _supported pairs_ into that slot, so the published
151,704 read as 742 and one column held two different quantities across the same
sheet.

Of the two options — restore the total under its own name and move the current
quantity elsewhere, or rename the column to say what it holds — the first was
taken, because the published name is correct for the quantity the caption
describes and because the sheet was already inconsistent with itself. The
supported-pair count moves to `n_support`, beside the `n_no_support` it pairs
with, and keeps its position in the column order; `total_indirect_assoc` is
inserted ahead of it and filled from the last year of
`temporal_drug_enrichment_full_chembl.csv`, which is the same
`chemblDrugEnrichment` call over the same full L2G evidence, so the constant is
derived rather than typed in. The full-dataset row now reads
`total_indirect_assoc` 151,704 and `n_support` 742.

Nothing else moved. Every odds ratio, relative success, confidence interval, P
and count column is identical to the sheet as it stood before the change, and
`total_indirect_assoc` went from agreeing with the submitted workbook on 18 of
33 rows to all 33.

## The caption is the specification (2026-08-24)

Six sheets were reissued so their columns are the columns their caption in
`~/Projects/manuscript_gentropy/sections/extended_data.tex` lists, in that
order, under headers a reader can read. Only `01_supplementary_tables.ipynb`
changed; `03_l2g_tables.ipynb` was deliberately left alone (see ST13 below). No
`.tex` file was touched. The counts these sheets now assert were staged in
`tools/expected_numbers_candidates_st_captions.tsv` and were merged into
`tools/expected_numbers.tsv` on 2026-08-25 as sixteen `T2.*`, `T5.*`, `T14.*`,
`T15.*` and `T16.*` rows; the candidates file is deleted.
`01_supplementary_tables.ipynb` emits them to
`results/supplementary_tables.json`, so all sixteen carry a verdict, and all
sixteen pass.

Two of the seventeen candidate rows did not survive as they stood. **`T2.01`**
is the only one whose `published` value differs from `expected`: the submitted
`tab:st2` caption read "The 31 lead variants (out of 135 with lead
vPS\,$\geq$\,10)", so 31 against the current 37 — a manuscript edit owed, and
the caption at `HEAD` already carries it. The same caption gained the
representative count, `T2.02` = 18, which the submitted caption did not carry at
all; it is registered with `published` equal to `expected` because there is no
submitted value for it to differ from. `tab:st5`'s three counts are unchanged
since submission, and `tab:st14`, `tab:st15` and `tab:st16` were added after
submission, so those twelve rows owe nothing. **`T16.02`**, the 2,757 genes
differing under the legacy therapeutic-area hierarchy, was not merged:
`THERAPEUTIC_AREAS_LEGACY` was deleted, so nothing in the pipeline computes it
and nothing can. It is recorded here as a historical number rather than
registered as a pending one.

| sheet                              | shape       | verdict                                                                      |
| ---------------------------------- | ----------- | ---------------------------------------------------------------------------- |
| ST2 discordant variants            | 37 x 14     | reordered and renamed; three columns kept past the caption, five dropped     |
| ST5 ChEMBL T-I pairs               | 37,377 x 13 | gene symbol and disease name added (R2-MJ-17); all three caption counts pass |
| ST13 effector gene list            | 12,274 x 5  | **not rebuilt** — the file is a different artefact from the caption          |
| ST14 gene-disease with gPS         | 36,858 x 7  | reordered and renamed; reconciled against R1.31 in the notebook              |
| ST15 cluster membership            | 42,918 x 6  | reordered, renamed, sorted on `Cluster ID`; all five controls pass           |
| ST16 therapeutic-area distribution | 25 x 7      | columns already matched; gPS counts are now integers, not `240.0`            |

`ST6_drug_target_enrichment.csv` also moved, and not by intent: the notebook
copies `drug_enrichment_subsets_vs_full_l2g.csv` through unchanged, and that
derived table has gained a `z_p_value` column since the sheet was last written.
Every value the sheet already carried is byte-identical; the sheet had simply
gone stale against its own input. Re-running the notebook is what refreshed it.

### ST2 — which disease count is which

The caption's "number of associated diseases" is **`signedLeadVPS`**, the
amended sign-gated lead_vPS the selection is made on, shipped as
**`Number of associated diseases`**. Three disease counts sat on the old sheet
under headers no reader could tell apart, and the other two are now dropped
rather than renamed: `leadVPS` (the first redefinition) and `uniqueDiseases`
(the published column). `leadDirectionalConcordance`, `uniqueTherapeuticAreas`,
`betaSignConcordance` and `therapeuticAreaNames` go with them — none entered a
selection, and no manuscript number is read off this sheet.

Three columns are kept past the caption list, each because a quoted count is
read off it: `Cluster representative` (the 18 rows the main text is computed
over), `Cluster ID` (what makes that flag checkable, and what marks the six
multi-row loci) and `Ensembl gene ID(s)` (the 37 and 21 distinct-gene counts of
R4.19 are over ids, not symbols).

Controls, all asserted on the written sheet: **37 rows**, **18 with
`Cluster representative` true**, every row at
`Number of associated diseases >= 10` and `Directional concordance <= 0.8`.
Range is 10-71 and 0.5333-0.8000, three rows sitting exactly at 0.8.

### ST5 — R2-MJ-17 in full

`Gene symbol` joins from the release target index and `Disease name` from the
release disease index — the two sources ST7, ST14 and ST15 already use. Both
cover every row: 1,273 targets and 1,842 diseases, **0 unmatched**, so the join
adds no nulls and drops nothing. All three caption counts pass: **37,377 rows**,
**4,564 approved**, **242 of those with GWAS genetic support**.

### ST14 — 36,858 against the published 34,905

Both numbers are right; they count different things, and the notebook now prints
the reconciliation. This sheet is the **unrestricted** list of gene-disease
associations: 36,858 rows over 8,285 unique genes and 1,394 unique diseases.
R1.31 — the 34,905 quoted in Results 1 and the Discussion as the denominator of
the ancestry comparison — is the same pairs under the two restrictions that
paragraph is about:

| restriction                  | pairs  |
| ---------------------------- | ------ |
| none — this sheet            | 36,858 |
| common-variant evidence only | 36,213 |
| discovered by 2024           | 35,535 |
| both — R1.31                 | 34,905 |

1,953 pairs separate them: **645** have rare-variant evidence only, **1,323**
were first seen in 2025 only (a partial year in the release), and **15** fail
both. Neither restriction belongs on this sheet — the caption asks for the
complete list, and R2-MJ-6/R2-MJ-8 ask for it so a reader can filter on the
therapeutic-area count themselves.

### The ST16 hierarchy question is settled

The `tab:st16` caption carried a `TODO(revisit)` saying the hierarchy behind the
sheet reproduces the study-level therapeutic-area assignment but not the
gene-level columns behind gPS, and asking whether the Supplementary Table 9
ordering does. **It does.** Aggregating study-level therapeutic-area membership
to genes over `prioritised_genes_diseases` and comparing with the pre-refactor
`genes_therapeutic_areas` table, the Supplementary Table 9 ordering
(`primaryTherapeuticArea`) reproduces all 23 area columns and `totalStudies` for
**0 of 8,285 genes**. The legacy ordering did not: **2,757 genes differed**,
`genetic, familial or congenital disease` on 2,327 of them and
`immune system disease` on 1,478. The check is a cell in
`01_supplementary_tables.ipynb`, so it re-runs with the sheet; since the legacy
ordering was deleted on 2026-08-24 it has one arm, and asserts the 0.

That settles the open decision above in favour of reissuing ST16 rather than
pinning it back to the legacy column, and the `TODO(revisit)` marker can be
deleted. Against the workbook sheet built on 14 August
(`~/Projects/manuscript_gentropy/supplementary_tables/sheets/ST16_-_TA_distribution.xlsx`,
25 data rows x 7 columns), the CSV differs in exactly the way that decision
implies:

- **the same 7 columns and the same 25 rows**, and the same totals — 2,320
  qualifying disease terms, 1,394 gPS terms, 7,010 measurement terms
- **row order differs on two rows**: `genetic, familial or congenital disease`
  is second here and second-from-last there, mirroring where the two hierarchies
  put it
- **18 of the 22 area rows carry different counts.** The two largest moves are
  the two areas the hierarchies disagree about:
  `genetic, familial or congenital disease` 8/2 there against 180/121 here, and
  `immune system disease` 111/75 there against 29/15 here.
  `cancer or benign tumor` (350/240), `pancreas disease` (12/10),
  `psychiatric disorder` (15/9) and `sign or symptom` (43/22) are unchanged
- formatting: the xlsx stores an unrounded percentage on one row
  (`disorder of ear`, `0.6899999999999999`), and the CSV used to write both
  count columns as floats because the measurement row has no gPS count. They are
  nullable integers now, so a reader sees `240` and not `240.0`

**The one divergence this used to record is fixed — 2026-08-24.**
`01-data-preparation/03_therapeutic_areas` built the `paper.TA_COLUMNS` one-hot
block from `mappedTherapeuticAreasLegacy`, so the per-area count columns
`06_gene_level_table` summed onto `gene_table` were legacy-ordered, and they
were the 2,757-gene half of the comparison above. It now builds them from
`mappedTherapeuticAreas`, and that half of the comparison is 0 of 8,285 too.

The cost is that `study_therapeutic_areas` no longer matches the pre-refactor
`gwas_w_therapeutic_areas` on those columns: **2,182 of 100,526 studies
differ**, led by `geneticFamilialOrCongenitalDisease` (1,600) and
`immuneSystemDisease` (993). That cannot be avoided, because the pre-refactor
pipeline was itself inconsistent — its study table was legacy-ordered and its
gene table was not — so no single hierarchy reproduces both. The study-level
control in notebook 03 is now a report rather than an assertion, and the
gene-level control here is the one that holds.

It never propagated in any case. Grepping for the per-area column names and
`totalStudies` outside the notebook that writes them returns **no consumer**:
ST7's 2-5 therapeutic-area window, ED Fig 7 and every gPS number read
`uniqueDiseases` and `uniqueTherapeuticAreas`, which come from
`mappedTherapeuticAreas` and match the published gene-level table for **0 of
8,285 genes differing** — which is also why the caption's worry that this
"affects ED Fig 7, tab:st8 and the 2-5 TA definition" does not hold.

**ST2 moved.** Repointing `07_variant_features` off the legacy alias moves
`Number of associated therapeutic areas`: 19 of the 37 rows go up by one
(`6_90267049_G_A` up by three, 4 -> 7), column total 231 -> 252. The row set and
every other column are unchanged, as are ST5, ST7, ST9, ST14, ST15, ST16 and
ST17 — verified sheet-by-sheet against the pre-change CSVs.

### ST13 is not the effector gene list

**Verdict: a different artefact entirely — the L2G training locus table.**
`sheets/ST13_effector_gene_list.csv` is the positive half of
`data/l2g_training_set/20250625_gentropy_paper_v1` exploded on `diseaseIds`:
12,274 rows of `diseaseId, geneId, studyId, studyLocusId, variantId`, one per
locus. That parquet's whole schema is those five fields plus `goldStandardSet`.
It has no `source` and no `confidence`, and it is a _downstream_ product — the
gold standard positives are the effector gene list after it has been joined onto
credible sets and the L2G feature matrix. So the caption's de-duplicated
EFO-gene list, with contributing source and confidence level, cannot be built
from it and option 1 and option 2 are both out.

The scale says the same thing. De-duplicated, those 12,274 locus rows collapse
to **1,704 EFO-gene pairs over 390 genes and 291 diseases** — what survived
being matched to a credible set. `01_EGL_preparation.ipynb` counts its combined
list at **42,288 pairs** before any locus join. A sheet published under this
caption would be short by a factor of 25.

**The effector gene list does exist, and it is not in this pipeline.** It is
built by `playground/EGL_and_training_set/01_EGL_preparation.ipynb` and written
to
`gs://genetics-portal-dev-analysis/yt4/2506_release/training_set/20250625_EGL_2506_0.95_otg_chembl.parquet`,
which `02_training_set.ipynb` then reads as its `sgl` input. Nothing under
`data/` holds it — `find data -iname '*EGL*'` is empty — so it is GCS-only and
the local pipeline cannot rebuild the sheet without that read.

The three sources are exactly the caption's, and each branch is visible in that
notebook:

| branch                                          | selection                                                                                                                                               | pairs      | input                        |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------------------------- |
| previous Open Targets Genetics gold standards   | `otg_gs_230511.json`, `gold_standard_info.highest_confidence in (High, Medium)` — 629 High, 650 Medium — `trait_info.ontology` exploded                 | 812        | **GCS only**                 |
| ChEMBL Phase 3/4                                | release `evidence/sourceId=chembl`, `clinicalPhase in (3, 4)`                                                                                           | 25,685     | `data/25.06/output/evidence` |
| high-confidence platform associations           | release `evidence`, `score >= 0.95`, `datasourceId in (eva, uniprot_variants, gene2phenotype, genomics_england, clingen, uniprot_literature, orphanet)` | 16,371     | `data/25.06/output/evidence` |
| union, de-duplicated on `(targetId, diseaseId)` |                                                                                                                                                         | **42,288** |                              |

Two facts matter for the response letters:

1. **Even the EGL parquet cannot fill the caption's last two columns.** The
   three branches are combined as
   `exploded_df.union(sgl).union(sgl0).distinct()` and then
   `dropDuplicates(["targetId", "diseaseId"])`, so the artefact is two columns
   wide — `diseaseId`, `targetId`. `datasourceId` and the High/Medium confidence
   label are dropped by the union, not carried through. Contributing source and
   confidence would have to be **reconstructed** by re-running each branch with
   a source tag and outer-joining the three.
2. **Two of the three branches are reconstructible locally, the third is not.**
   The ChEMBL and high-confidence-platform branches both read
   `data/25.06/output/evidence`, which is on disk. The Open Targets Genetics
   gold standards branch reads `otg_gs_230511.json` from GCS, and that file is
   the only carrier of the `High`/`Medium` confidence label the caption asks
   for. Note also that `europepmc` literature evidence is prepared in the
   notebook and then deliberately **not** unioned in (the cell is marked
   `LITERATURE - SKIP`), consistent with the caption not listing it.

Not regenerated, and the caption not rewritten to fit. R1-mn-10 and R2-MJ-9 turn
on which of those two the author wants: fetch the GCS json and build the sheet
with a reconstructed source and confidence column, or narrow the caption to the
two columns the artefact actually carries.

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

# Supplementary tables compiled by hand

Some published sheets were **assembled manually**, not produced by any script
here. A missing builder for those is **not** a reproducibility gap; do not
reverse-engineer one without asking.

They are handled two ways:

| Sheet                        | Where        | Treatment                                                                                                                                                                                                                                     |
| ---------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ST8 GSEA                     | `../assets/` | **static asset**, shipped as submitted. Written by hand by Polina; not rebuilt, not recomputed                                                                                                                                                |
| ST16 colocalisation overlap  | `../assets/` | **static asset**, shipped as submitted. Recomputation disagrees with subtable 1 throughout and with subtable 2's per-assay columns, so the definitions are treated as unrecovered. Two Supplementary Results numbers depend on it — see below |
| ST13 subgroup analysis       | here         | hand-made layout, every number recomputed; **reissued 2026-10-20** by `build_st8.py` — see below                                                                                                                                              |
| ST15 fine-mapping statistics | not copied   | sheet stays in the manuscript tree; `../04_fine_mapping_numbers.ipynb` reproduces **54 of 54** checkable cells                                                                                                                                |

## Refreshing the numbers

```
tools/run_chapter.sh chapters/06-supplementary-tables 02
```

`../02_manual_table_numbers.ipynb` recomputes every count in ST13 and ST16 from
the pipeline and writes one CSV per subtable to `refreshed/`, alongside a
`*_vs_published.csv` comparison.

**It does not touch the spreadsheets.** Editing them is a deliberate manual
step: read the comparison, decide whether the change is meant, then paste. That
keeps a hand-made sheet from being silently overwritten by a number nobody has
looked at.

## Known differences, as of 2026-10-20

Read this before pasting anything. **ST13 now reproduces and has been reissued;
ST16 does not, and its recipe should be treated as unverified.**

### ST13 — reissued 2026-10-20 on the Supplementary Table 14 hierarchy

`build_st8.py` writes `refreshed/ST13_-_subgroup_analysis.xlsx` from the four
refreshed CSVs, keeping the published layout (title, description, one note per
column, four subtables). That file was copied over
`supplementary_tables/sheets/ST13_-_subgroup_analysis.xlsx` in the manuscript
repository and the merged workbook rebuilt with `make xlsx`. The copy of the
submitted sheet in this folder is kept as the record of what was published.

```
uv run --with openpyxl python chapters/06-supplementary-tables/manual/build_st8.py
```

- **Tables 1 and 3 (by ChEMBL level-1 class) reproduce exactly** — all 45 cells
  of each, 14,532 and 1,001. The recovered definition: a target carrying several
  level-1 classes (265 targets do) is counted in **every** one. The notebook
  previously took the first class in the list, which gave 13,112 and 965 and the
  "Enzyme -29, Unclassified -5, Secreted -2" residual recorded here before;
  those are exactly the multi-class targets. The totals are therefore class
  memberships: 13,112 and 965 distinct pairs carry a class. Genes with no ChEMBL
  class are dropped, not put under `Unclassified protein`.
- **Tables 2 and 4 (by therapeutic area) are new.** The published ones used the
  old hierarchy, which counted a disease in every area it reached (totals 54,307
  and 4,012) and carried `Animal disease` (EFO_0005932) and `Medical procedure`
  (EFO_0002571). Now each disease has the one area Supplementary Table 14
  assigns, so the totals are the distinct pairs: **36,858** (Supplementary
  Table 7) and **2,734** (Supplementary Table 12). Rows are the 22 disease areas
  in ST14 order plus `other (no area root)`; `measurement` has no row because
  the prioritised pairs are disease-only.
- **Column note corrected.** The description said subtables 1-2 were "all
  PAV-supported associations"; they are all prioritised gene-disease pairs.
- **Display bug fixed.** Until 2026-10-20 every row of the Table 2/4 CSVs was
  labelled `other (no area root)`, because the id-to-name map was inverted. The
  counts were right.
- **The subgroup is biased toward target classes**, which Results 6 says it is
  not. `refreshed/ST13_bias_by_target_class.csv` and
  `ST13_bias_by_therapeutic_area.csv` hold the shares, Fisher P and BH FDR; the
  numbers are in `chapters/06-supplementary-tables/README.md`.

**A parse trap worth knowing.** An earlier version of this notebook read ST12's
`diseaseIds` column with `ast.literal_eval`. Pandas writes a numpy array to CSV
as `"['A' 'B' 'C']"` — space separated, no commas — and Python parses that as
_implicit string concatenation_, yielding the single token `'ABC'`. Every
multi-disease credible set silently collapsed to one, costing about 10%. ST12
now carries an exploded `diseaseId` column, so no parsing is needed.

### ST16 — subtable 1 does not reproduce; subtable 2 does, in its totals

- **Subtable 1** recomputes uniformly **higher** than published: GWAS eCAVIAR
  48,998,746 against 41,751,740, total 75,407,692 against 61,484,864. The
  row-by-row ratio is not constant (GWAS 0.85x, eQTL 0.77x, pQTL 0.61x), so it
  is not one filter applied to the release; an earlier data vintage is the
  likeliest explanation, but it has not been identified.
- **This subtable is where the Supplementary Results percentages come from.**
  Its totals give 41,398,927 / 61,484,864 = 67.3% for eCAVIAR and 24,536,009 /
  31,167,732 = 78.7% for COLOC, which is the published "67% ... and 79%" of
  Supplementary Results 2 exactly. The release gives 68.0% and 79.3%. S2.01 and
  S2.02 are therefore marked `precomputed` in `tools/expected_numbers.tsv` and
  recorded in
  `chapters/03-analysis-supplementary/02_systematic_colocalisation.ipynb` — they
  cannot be closed until this sheet's universe is.
- **Subtable 2 agrees where it is comparable.** Its `All` column, 285,229
  credible sets and 14,026 unique genes, is exactly what the pipeline computes
  for qualifying credible sets with a protein-coding, non-_trans_ molQTL
  colocalisation. The per-assay columns disagree in both directions — eQTL
  268,059 against 253,643, pQTL 123,913 against 43,038 (nearly 3x), sQTL 113,789
  against 167,703 (lower) — and the sheet's own header calls its pQTL column
  _cis_-pQTL, which the recomputation does not restrict to. So the disagreement
  here is per-assay scoping, not the same problem as subtable 1.
- **Do not paste ST16 numbers into the sheet** until that is resolved. The
  notebook is useful as a starting point, not as an answer.

### ST15 — every checkable cell reproduces (54 of 54)

All six checkable columns are exact for all nine data sources: number of valid
studies, unique EFO/genes, unique regions, number of CSs, % of binary traits,
and % of CS with SNP PIP > 0.9.

Three definitions had to be recovered, all now recorded in the notebook:

- The **curated GWAS Catalog row includes the 410 studies whose `hasSumstats` is
  null**, all PICS fine-mapped. 17,781 + 410 = 18,191, the published figure
  exactly.
- **PICS credible sets carry no `region`**, so the published sheet reports the
  credible-set count in the regions column for that row.
- **"Number of unique EFO/genes" counts distinct `diseaseIds` arrays, not
  distinct disease terms** — trait _combinations_. From `stats_from_list()` in
  `chapters/_legacy/02-analysis/01-descriptions-numbers/01_descriptive_numbers.ipynb`.
  Exploding first gives 6,348 / 3,393 / 1,001 against 7,047 / 3,698 / 974;
  FinnGen moving the opposite way to the other two is the giveaway, since
  counting combinations is not ordered against counting members.

"Original number of studies before ingestion" is the one column that cannot be
recomputed: it counts studies in each source's own index before ingestion, which
needs the per-source `study_index` buckets listed in `GAPS.md`.

### ST8

The gene sets are public and still available — `Reactome_Pathways_2024` (2,105
sets) and `KEGG_2026` (352) from Enrichr, and 2,105 + 352 = 2,457 is exactly the
published row count. Recorded because it was previously listed as a blocked
input; it does not mean the table should be rebuilt.

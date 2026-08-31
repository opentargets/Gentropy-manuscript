# Supplementary tables compiled by hand

Some published sheets were **assembled manually**, not produced by any script
here. A missing builder for those is **not** a reproducibility gap; do not
reverse-engineer one without asking.

They are handled two ways:

| Sheet                        | Where        | Treatment                                                                                                                                                                                                                                     |
| ---------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ST3 GSEA                     | `../assets/` | **static asset**, shipped as submitted. Written by hand by Polina; not rebuilt, not recomputed                                                                                                                                                |
| ST11 colocalisation overlap  | `../assets/` | **static asset**, shipped as submitted. Recomputation disagrees with subtable 1 throughout and with subtable 2's per-assay columns, so the definitions are treated as unrecovered. Two Supplementary Results numbers depend on it — see below |
| ST8 subgroup analysis        | here         | hand-made sheet, but every number in it **is** recomputable — see below                                                                                                                                                                       |
| ST10 fine-mapping statistics | not copied   | sheet stays in the manuscript tree; `../04_fine_mapping_numbers.ipynb` reproduces **54 of 54** checkable cells                                                                                                                                |

## Refreshing the numbers

```
tools/run_chapter.sh chapters/06-supplementary-tables 02
```

`../02_manual_table_numbers.ipynb` recomputes every count in ST8 and ST11 from
the pipeline and writes one CSV per subtable to `refreshed/`, alongside a
`*_vs_published.csv` comparison.

**It does not touch the spreadsheets.** Editing them is a deliberate manual
step: read the comparison, decide whether the change is meant, then paste. That
keeps a hand-made sheet from being silently overwritten by a number nobody has
looked at.

## Known differences, as of 2026-08-19

Read this before pasting anything. **None of the recomputed blocks currently
reproduces its published values exactly.** ST8 is close and the cause is
understood; ST11 is not, and its recipe should be treated as unverified.

### ST8 — Tables 3/4 nearly exact; Tables 1/2 use a wider universe

- **Tables 3 and 4** (PAV + 2-5 therapeutic areas): 965 pairs against the
  published 1,001. **11 of the 15 target classes now match exactly**; the
  residual sits in Enzyme (-29), Unclassified protein (-5) and Secreted protein
  (-2), so it is a specific set of genes rather than a systematic offset. ST7
  itself now reproduces exactly, so this is no longer inherited.
- **Tables 1 and 2** (all nominated genetic evidence): 13,112 pairs against
  14,532, with the shape right throughout — Enzyme 5,631 / 827 / 1,115 against
  5,857 / 835 / 1,151. Unresolved; the published universe appears slightly wider
  than `prioritised_genes_diseases`.
- **The therapeutic-area list is not the pipeline's.** The published sheet
  includes `Animal disease` (EFO_0005932) and `Medical procedure` (EFO_0002571),
  which are not among the 23 areas in `paper.THERAPEUTIC_AREAS`. The
  recomputation uses the pipeline's hierarchy and reports an
  `other (no area root)` row instead, so Tables 2 and 4 will not align row for
  row.

**A parse trap worth knowing.** An earlier version of this notebook read ST7's
`diseaseIds` column with `ast.literal_eval`. Pandas writes a numpy array to CSV
as `"['A' 'B' 'C']"` — space separated, no commas — and Python parses that as
_implicit string concatenation_, yielding the single token `'ABC'`. Every
multi-disease credible set silently collapsed to one, costing about 10%. ST7 now
carries an exploded `diseaseId` column, so no parsing is needed.

### ST11 — subtable 1 does not reproduce; subtable 2 does, in its totals

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
- **Do not paste ST11 numbers into the sheet** until that is resolved. The
  notebook is useful as a starting point, not as an answer.

### ST10 — every checkable cell reproduces (54 of 54)

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

### ST3

The gene sets are public and still available — `Reactome_Pathways_2024` (2,105
sets) and `KEGG_2026` (352) from Enrichr, and 2,105 + 352 = 2,457 is exactly the
published row count. Recorded because it was previously listed as a blocked
input; it does not mean the table should be rebuilt.

# Distribution of diseases across therapeutic areas

> **R3-mn-4** — "The manuscript does not appear to include a dedicated section
> describing how traits are distributed across therapeutic areas. A summary of
> disease or trait distribution across therapeutic areas would be useful.
> Because multiple clusters of related traits may exist within a therapeutic
> area, and because some traits may show high genetic correlation across
> therapeutic areas, the authors may also consider whether clustering traits by
> genetic correlation would provide additional evidence supporting the
> assumption of independence between therapeutic areas."

This folder answers the **first** part — the missing distribution table. The
genetic-correlation part is handled separately in `../ta-independence/`.

## The table — `ta_distribution_supplementary-r1.csv`

One row per therapeutic area, counting **diseases**, over the two trait
universes the manuscript uses:

| universe               | what it is                                                                                                              | size                                            |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| **qualifying dataset** | ontology terms used by studies that passed QC and entered the analyses                                                  | 2,320 disease terms (+ 7,010 measurement terms) |
| **gPS disease list**   | terms with ≥ 1 qualifying credible set carrying an L2G-prioritised gene — the universe gPS and gps_TA are computed over | 1,394 terms                                     |

Columns: `therapeutic_area`, `root_id`, `trait_class`, `n_diseases_qualifying`,
`n_diseases_gps`, `pct_qualifying`, `pct_gps`.

| Therapeutic area                             | Qualifying |     % |  gPS list |     % |
| -------------------------------------------- | ---------: | ----: | --------: | ----: |
| _(measurement)_                              |    _7,010_ |     — |         — |     — |
| other (no area root)                         |        586 | 25.26 |       303 | 21.74 |
| cancer or benign tumor                       |        350 | 15.09 |       240 | 17.22 |
| nervous system disease                       |        175 |  7.54 |        90 |  6.46 |
| cardiovascular disease                       |        161 |  6.94 |       116 |  8.32 |
| infectious disease                           |        142 |  6.12 |        53 |  3.80 |
| disorder of visual system                    |        117 |  5.04 |        85 |  6.10 |
| immune system disease                        |        111 |  4.78 |        75 |  5.38 |
| musculoskeletal or connective tissue disease |         97 |  4.18 |        66 |  4.73 |
| gastrointestinal disease                     |         95 |  4.09 |        59 |  4.23 |
| reproductive system or breast disease        |         77 |  3.32 |        35 |  2.51 |
| injury, poisoning or other complication      |         65 |  2.80 |        39 |  2.80 |
| integumentary system disease                 |         56 |  2.41 |        36 |  2.58 |
| respiratory or thoracic disease              |         49 |  2.11 |        34 |  2.44 |
| urinary system disease                       |         45 |  1.94 |        32 |  2.30 |
| sign or symptom                              |         43 |  1.85 |        22 |  1.58 |
| nutritional or metabolic disease             |         37 |  1.59 |        29 |  2.08 |
| endocrine system disease                     |         27 |  1.16 |        21 |  1.51 |
| pregnancy or perinatal disease               |         18 |  0.78 |        11 |  0.79 |
| hematologic disease                          |         18 |  0.78 |        15 |  1.08 |
| disorder of ear                              |         16 |  0.69 |        12 |  0.86 |
| psychiatric disorder                         |         15 |  0.65 |         9 |  0.65 |
| pancreas disease                             |         12 |  0.52 |        10 |  0.72 |
| genetic, familial or congenital disease      |          8 |  0.34 |         2 |  0.14 |
| **TOTAL (disease side)**                     |  **2,320** |   100 | **1,394** |   100 |

## The mapping rule is the manuscript's own, and that is verified

One area per disease: the first `therapy_area_hierarchy` root found in the
term's `ancestors`, in the dictionary's own order, else `other` — verbatim from
`chapters/01-data-preparation/04_qualifying_dataset_generation.ipynb`.

The term → area map is **not persisted anywhere** — notebook 04 builds it inside
a Spark UDF closure and writes out only the study-level
`mappedTherapeuticAreas`. So it has to be rebuilt here. The notebook proves the
rebuild is the real thing rather than a lookalike: re-deriving each study's
`mappedTherapeuticAreas` from it reproduces the stored column for **all 15,730
qualifying studies**, and the assertion fails the notebook if it ever stops
doing so.

Because the rule assigns exactly one area per term, the rows are a partition:
the 22 areas plus `other` sum to 2,320 and to 1,394 exactly, also asserted.

**Reproduced as-is, including its bug.** The rule tests `ancestors` only, and a
term is not its own ancestor, so a GWAS annotated _directly_ to an area root is
labelled `other`. 13 terms are affected — cardiovascular disease, nervous system
disease, urinary system disease, infectious disease, gastrointestinal disease,
immune system disease, hematologic disease, endocrine system disease,
psychiatric disorder, disorder of ear, disorder of visual system, pancreas
disease, sign or symptom. They are listed at the end of the notebook. A
descriptive table should match the paper and disclose the defect, not silently
repair it; the repair belongs in `MAPPING_REVIEW.md`.

## What to say from it

1. **All 22 disease areas are populated in both universes** — every area has at
   least one qualifying term _and_ at least one term in the gPS list. The
   manuscript's "all 23 therapeutic areas" claim survives the restriction to the
   analysed subset, which is the specific thing the referee could not check.
2. **The distribution is uneven but not degenerate**
   (`ta_distribution_concentration-r1.csv`), over the 22 named areas: Gini
   0.478, Shannon evenness 0.875, largest area 20.2% of named-area diseases, top
   three 39.6%.
3. **Gene support does not skew the area mix.** The gPS list gives almost
   identical concentration — Gini 0.479, evenness 0.870, top three 40.9% — so
   restricting to gene-supported diseases is not selecting a narrower slice of
   medicine.
4. **A quarter of diseases carry no therapeutic area**, and this should be
   disclosed rather than buried in a residual row. See below.

## Why 586 diseases have no therapeutic area

None of them are missing from the ontology — all 586 have a row, they simply
descend from no area root (`ta_distribution_other_breakdown-r1.csv`):

| prefix | terms                         |
| ------ | ----------------------------- |
| HP     | 377                           |
| EFO    | 166                           |
| GO     | 24                            |
| OBA    | 15                            |
| MONDO  | 3 (the area roots themselves) |
| MP     | 1                             |

HP phenotype codes sit under `phenotype` rather than under any disease root by
construction, which is why they dominate. The full diagnosis, and what a fix
would be worth, is in **`MAPPING_REVIEW.md`** in this folder.

## Notebook

| Notebook                                 | Needs Spark | Runtime | What it does                                                                                                                                                                        |
| ---------------------------------------- | ----------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_trait_distribution_across_tas.ipynb` | no          | ~1 min  | rebuilds the area map and verifies it against the stored column, counts diseases per area for both universes, concentration statistics, term-level backing table, `other` breakdown |

```bash
cd chapters/06-review-r1/ta-distribution
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 01_trait_distribution_across_tas.ipynb
```

## Exports (all in `data/intermediate_files/`)

| File                                     | Contents                                                                     |
| ---------------------------------------- | ---------------------------------------------------------------------------- |
| `ta_distribution_supplementary-r1.csv`   | **the table** — diseases per therapeutic area and percentage, both universes |
| `ta_distribution_terms-r1.csv`           | one row per disease term (2,320): id, name, area, gPS-list membership        |
| `ta_distribution_concentration-r1.csv`   | Gini, Shannon evenness, largest and top-3 share, per universe                |
| `ta_distribution_other_breakdown-r1.csv` | the 586 unmapped terms by ontology prefix                                    |

## Companion

**`MAPPING_REVIEW.md`** — audit of the disease → therapeutic-area mapping
itself: six defects, the HP coverage problem, and what a crosswalk would
recover. Separate from this table, which only describes the mapping as it
currently stands.

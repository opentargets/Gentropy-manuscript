# Review of the disease → therapeutic-area mapping

Companion to `README.md`. Written while building the R3-mn-4 distribution table;
the referee is asking about this hierarchy, so its defects are in scope for the
response even though the table itself does not depend on all of them.

## The rule as implemented

`chapters/01-data-preparation/04_qualifying_dataset_generation.ipynb`, cells 4
and 7–8:

1. `therapy_area_hierarchy` — an ordered dict of 23 ontology roots. Dict order
   **is** the priority order.
2. Per **ontology term**: `primaryTherapeuticArea` = the first root in that
   order found in the term's `ancestors` list; `other` if none. **One area per
   term.**
3. Per **study**: `mappedTherapeuticAreas` = the deduplicated set of its
   `diseaseIds`' primary areas. A study therefore gets more than one area only
   when it carries more than one `diseaseId`, never because a single term sits
   under several roots.
4. `measurement` = `EFO_0001444 ∈ mappedTherapeuticAreas`; this flag splits the
   disease and measurement qualifying sets.
5. 23 one-hot columns + `other` per study, later summed over a gene's credible
   sets in
   `chapters/02-analysis/05-gene-level-ps/01_gene_level_pleiotropy.ipynb`, where
   `uniqueTherapeuticAreas` (= gps_TA) is the size of the union.

Verified: step 3 reproduces the stored `mappedTherapeuticAreas` for all 15,730
qualifying studies, exactly.

## Defects, worst first

### 1. The published gene-level TA columns were not produced by this priority order

`genes_therapeutic_areas.csv` cannot be regenerated from the current notebook.
Recomputing gps_TA from `l2g_diseases_full-r1.csv` + the stored
`mappedTherapeuticAreas` matches the published `uniqueTherapeuticAreas` for only
**87.3%** of the 8,285 genes, and the per-gene credible-set count for **78.7%**.
One column is off by two orders of magnitude:

|                                                                 | published `genes_therapeutic_areas.csv` | recomputed, current order |
| --------------------------------------------------------------- | --------------------------------------- | ------------------------- |
| genes with ≥ 1 `geneticFamilialOrCongenitalDisease` association | **2,338**                               | **17**                    |

`OTAR_0000018` is 21st of 22 in the current order, so it only ever catches terms
nothing else claims. Moving it to **second place, immediately after cancer**,
cuts the total absolute error across all 23 columns from 4,938 to 1,498 and
makes eight columns match the published table _exactly_ (genetic/familial 2,338,
cancer 2,045, pancreas 1,638, nutritional/metabolic 376, reproductive 224,
injury 161, pregnancy 108, and `other` to within 4%).

Residual mismatches under that reorder point to a few more rank changes — immune
+88%, ear +115%, urinary −46%, integumentary −31%, musculoskeletal −17% — plus
the ~21% drift between the credible-set snapshot behind
`genes_therapeutic_areas.csv` and `l2g_diseases_full-r1.csv`.

**Consequence.** Every TA-stratified result — gps_TA, ED Fig. 7's leave-one-out
by therapeutic area, ST8's subgroup analysis, and the "2–5 therapeutic areas"
definition behind OR = 10.3 — rests on a priority order that the repository no
longer contains. This is the single thing worth fixing before the revision goes
back: not because the numbers are necessarily wrong, but because they cannot
currently be reproduced from the code, and a referee or a reader who tries will
get 17 where the paper says 2,338.

**Fix:** recover the order that was actually run (fit it against the published
columns, or find it in the history of the notebook), restore it in cell 4 with a
comment saying it is load-bearing, and add an assertion in
`01_gene_level_pleiotropy.ipynb` that the regenerated columns match the
published ones.

### 2. `other` is 24.7% of disease terms and is counted as a therapeutic area

`uniqueTherapeuticAreas` is the count of non-zero area columns **including
`other`** (verified: exact for all 8,285 genes). So `other` is the 23rd
therapeutic area in every pleiotropy statement the paper makes.

- **2,730 of 8,285 genes (33%)** have at least one association in `other`
- **330 genes** have `other` as their _only_ area, i.e. gps_TA = 1 purely by
  residual

This cuts both ways and neither is defensible. A gene associated with ischemic
stroke (HP_0002140) and hypercholesterolemia (HP_0003124) scores gps_TA = 1 when
it should be 2; a gene associated with tinnitus, myopia and sepsis scores 1 when
it should be 3. Meanwhile a gene with one association to "medical procedure" and
one to breast cancer scores 2 as if it were cross-area pleiotropic.

Since the OR = 10.3 headline is defined on the band gps_TA ∈ [2, 5], mis-scoring
at exactly this end of the scale is not cosmetic.

**Fix, minimum:** exclude `other` from `uniqueTherapeuticAreas` and report the
change. **Fix, better:** shrink `other` first — see the next point.

### 3. Most of `other` is trivially mappable

None of the 574 unmapped qualifying disease terms are missing from the ontology;
they simply descend from no area root. Open Targets' own `therapeuticAreas`
column labels all 574: **447 `phenotype` (EFO_0000651), 92 `biological_process`
(GO_0008150), 35 `medical procedure` (EFO_0002571)**.

The 447 phenotype terms are overwhelmingly HP codes with **full HP organ-system
ancestry already present in `disease.parquet`** — HP_0002140 "Ischemic stroke"
carries HP_0001626 (cardiovascular) and HP_0000707 (nervous); HP_0000360
"Tinnitus" carries HP_0000598 (ear); HP_0100806 "Sepsis" carries HP_0002715
(immune).

A ~20-line crosswalk from HP organ-system roots to therapeutic areas rescues
**344 of 574 terms (60%)** and **1,769 of 2,754 study–term links (64%)**,
without touching the EFO/MONDO side:

| would land in    | terms |     | would land in            | terms |
| ---------------- | ----- | --- | ------------------------ | ----- |
| nervous system   | 68    |     | integumentary            | 30    |
| gastrointestinal | 49    |     | head/neck                | 29    |
| cardiovascular   | 47    |     | respiratory              | 17    |
| immune           | 44    |     | ear                      | 8     |
| visual           | 37    |     | perinatal                | 8     |
| metabolic        | 33    |     | breast, endocrine, other | 4     |
| hematologic      | 32    |     |                          |       |

The 230 that survive are genuinely unmappable and should stay in `other`:
`EFO_0002571` medical procedure (59 studies), `GO_0009410` response to
xenobiotic stimulus (35), `EFO_0010130` health study participation (20),
`EFO_0004318` smoking behavior (15). That is a residual bucket worth defending;
574 is not.

**Fix:** add the HP crosswalk, then handle GO and `medical procedure` by
exclusion rather than by burying them in a bucket that counts as an area.

#### HP terms are effectively unmappable under the current rule

Mapping rate over the 2,320 qualifying disease terms, splitting
`sign or symptom` out because the pipeline's own comment says it is not a
therapeutic area:

| prefix   | terms   | → `other` | → `sign or symptom` only | → real clinical area | % real  |
| -------- | ------- | --------- | ------------------------ | -------------------- | ------- |
| MONDO    | 542     | 0         | 0                        | 542                  | **100** |
| Orphanet | 7       | 0         | 0                        | 7                    | 100     |
| DOID     | 1       | 0         | 0                        | 1                    | 100     |
| EFO      | 1,331   | 157       | 29                       | 1,145                | 86.0    |
| **HP**   | **399** | **377**   | **15**                   | **7**                | **1.8** |
| GO       | 24      | 24        | 0                        | 0                    | 0       |
| OBA      | 15      | 15        | 0                        | 0                    | 0       |
| MP       | 1       | 1         | 0                        | 0                    | 0       |

**No MONDO, Orphanet or DOID term ever lands in `other`. 98.2% of HP terms do.**
The 22 HP terms that escape mostly do not really escape — 15 land in
`sign or symptom` (Back pain, Abdominal pain, Chest pain, Fatigue, Sciatica…),
because `EFO_0003765` itself sits under `phenotype`. Only 7 HP terms of 399
reach a genuine clinical area, and they get there through cross-hierarchy
axioms: Paroxysmal supraventricular tachycardia, Mitral valve prolapse, Subdural
hemorrhage, Supraventricular tachycardia (cardiovascular), Esophagitis
(gastrointestinal), Non-accomodative esotropia (visual), Neurogenic bladder
(urinary).

#### What the crosswalk actually buys — and the direction of the bias

Applying an HP organ-system crosswalk (26 roots, same hierarchy priority)
recovers **383 of the 574 `other` terms (67%)** and **1,942 of 2,754 study–term
links (71%)**, fanning out over 16 real areas — nervous 62, cardiovascular 44,
gastrointestinal 38, musculoskeletal 38, immune 37, visual 37, metabolic 22,
urinary 22. Residual 191: medical procedure, response to xenobiotic stimulus,
health study participation, smoking behavior, the suicide items — genuinely not
diseases.

The effect on gps_TA is the informative part:

| definition                              | mean gps_TA | genes with gps_TA = 0 | genes in 2–5 band | genes changed |
| --------------------------------------- | ----------- | --------------------- | ----------------- | ------------- |
| **current** (`other` counts as an area) | **2.423**   | 0                     | **4,019**         | —             |
| `other` excluded entirely               | 2.109       | 308                   | 3,655             | 2,608         |
| HP crosswalk, `other` still counts      | 2.407       | 0                     | 3,935             | 836           |
| HP crosswalk + `other` excluded         | 2.255       | 85                    | 3,814             | 1,557         |

The crosswalk barely moves the mean (2.423 → 2.407) but changes 836 genes — and
**503 of them _lose_ a therapeutic area while only 333 gain one**. That is the
diagnosis: for most genes the recovered area is one the gene already had, so
`other` was acting as a **phantom 23rd area**. The bias from the residual bucket
is **upward**, not downward — it inflates apparent cross-area pleiotropy rather
than hiding it.

Note also that 308 genes have gps_TA = 0 if `other` is simply dropped without
the crosswalk (every one of their diseases is unmapped); with the crosswalk
first, only 85 do. That is the argument for doing the two fixes together rather
than just excluding `other`.

_(These crosswalk numbers were computed interactively and are not yet in a
notebook — see the closing section.)_

### 4. Root terms are assigned to `other`

The UDF tests `root in ancestors[term]`, and a term is not its own ancestor. So
a GWAS annotated directly to a therapeutic-area root is labelled `other`:

| term                               | studies | assigned                   |
| ---------------------------------- | ------- | -------------------------- |
| EFO_0000319 cardiovascular disease | 47      | `other`                    |
| EFO_0000618 nervous system disease | 44      | `other`                    |
| EFO_0009690 urinary system disease | 43      | `other`                    |
| EFO_0003765 sign or symptom        | 39      | `other`                    |
| EFO_0005741 infectious disease     | 26      | `other`                    |
| + 8 more roots                     |         |                            |
| EFO_0009605 pancreas disease       | 12      | **endocrine** (its parent) |

**270 qualifying studies**, 13 terms. Small, but it is a one-character bug —
`term == root or root in ancestors[term]` — and it is embarrassing in exactly
the place the referee is looking. This notebook's table already applies the
corrected rule; upstream does not.

### 5. The priority order silently decides two-thirds of the assignments

Only **33.7%** of qualifying disease terms descend from exactly one area root.
For the other 66% the answer is whatever the dict order says, and the dict has
no stated rationale. The distortion is large and systematic — areas near the
bottom of the list are starved:

| area                                  | terms under `_any` | terms under `_primary` | retained |
| ------------------------------------- | ------------------ | ---------------------- | -------- |
| cancer or benign tumor (rank 1)       | 350                | 350                    | 100%     |
| infectious disease (rank 2)           | 150                | 143                    | 95%      |
| gastrointestinal (rank 9)             | 221                | 96                     | 43%      |
| endocrine (rank 12)                   | 139                | 27                     | 19%      |
| psychiatric (rank 19)                 | 122                | 16                     | 13%      |
| genetic/familial/congenital (rank 21) | 216                | 8                      | 4%       |

Cancer-first is at least clinically defensible (oncology is a real therapeutic
area, and every organ cancer belongs to it). Ranks 9–21 are not obviously
anything.

There is a deeper issue: **first-match-wins makes gps_TA order-dependent in a
way that a "number of distinct areas" metric should not be.** Under the current
order a gene hitting colorectal cancer and gastric cancer scores 1; under a
gastrointestinal-first order it also scores 1; but a gene hitting colorectal
cancer and Crohn's disease scores 2 under the first and 1 under the second. The
metric is stable only because the order is frozen — which, per defect 1, it is
not.

**Fix, in ascending order of work:**

- **document** the order and state that it is load-bearing (mandatory regardless
  of anything else);
- **report gps_TA both ways** — first-match and full multi-membership — as a
  sensitivity analysis. Full multi-membership is order-free and reviewer-proof;
  if the OR = 10.3 result survives both, defect 5 stops being an attack surface;
- **replace first-match with most-specific-match** — assign the term to the area
  root with the fewest descendants among those it descends from. That is a
  principled tie-break rather than an editorial one, and it is a two-line
  change.

### 6. Minor: the measurement split is by flag, not by primacy

`measurement = EFO_0001444 ∈ mappedTherapeuticAreas`, so **2,025 studies**
carrying both a measurement term and a disease term go to the measurement side
whatever the disease term is. Conversely 136 measurement-side terms also descend
from a disease root. Defensible as a conservative choice, but it should be
stated, and it is the mechanism behind the 52 UK Biobank questionnaire items
that `../ta-independence/` found sitting in the _non_-measurement bucket.

#### The two candidate remaps, side by side

Coverage of the 2,320 qualifying disease terms under (A) the current rule, (B)
after merging duplicated ontology terms (`../ontology-duplicates/`), and (C)
after that plus the HP organ-system crosswalk:

| prefix          | A: terms / `other` / real area  | B: after duplicate merge        | C: + HP crosswalk                   |
| --------------- | ------------------------------- | ------------------------------- | ----------------------------------- |
| MONDO           | 542 / 0 / 542 (100%)            | 542 / 0 / 542 (100%)            | 542 / 0 / 542 (100%)                |
| EFO             | 1,331 / 157 / 1,145 (86.0%)     | 1,321 / 155 / 1,137 (86.1%)     | 1,321 / **101** / 1,190 (**90.1%**) |
| **HP**          | 399 / 377 / 7 (**1.8%**)        | 369 / 347 / 7 (1.9%)            | 369 / **33** / **317** (**85.9%**)  |
| GO / OBA / MP   | 40 / 40 / 0                     | 40 / 40 / 0                     | 40 / 39 / 1                         |
| Orphanet / DOID | 8 / 0 / 8                       | 8 / 0 / 8                       | 8 / 0 / 8                           |
| **TOTAL**       | **2,320 / 574 / 1,702 (73.4%)** | 2,280 / 542 / 1,694 (**74.3%**) | 2,280 / **173** / **2,058 (90.3%)** |

The duplicate merge absorbs 40 terms — **30 HP and 10 EFO, all into MONDO (30)
or EFO (10)** — and moves coverage by **+0.9 points**. The HP crosswalk moves it
by **+16 points** and cuts `other` from 574 to 173.

That is the clearest statement of the relative priority of the two problems:
**remapping duplicates is worth about a twentieth of what fixing the
therapeutic-area mapping is worth.**

## Recommendation

If only one thing gets done: **defect 1**, because a reviewer can hit it and the
paper's headline metric depends on it.

If three: **1**, then **4** (one character), then **2 + 3 together** (drop
`other` from gps_TA after the HP crosswalk shrinks it from 574 terms to ~230).
Then re-run gps_TA and check whether the 2–5 band and OR = 10.3 move — that
check is the actual deliverable, not the mapping change itself.

Defect 5 is best answered in the response letter with a sensitivity analysis
rather than by changing the pipeline this late.

## Where each number here comes from

All computed against `data/25.06/output/disease/disease.parquet`,
`data/intermediate_files/qualifying_gwas_studies`, `l2g_diseases_full-r1.csv`
and `genes_therapeutic_areas.csv`. The distribution numbers are in the exports
listed in `README.md`; the defect-1, defect-2 and defect-3 quantifications were
done interactively and are **not yet in a notebook** — they should be codified
before any of them is quoted in the response letter.

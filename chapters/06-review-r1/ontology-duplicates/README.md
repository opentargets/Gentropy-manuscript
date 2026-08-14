# Impact of duplicated ontology terms on this manuscript

**Problem 1 of two.** The same disease can exist in the Open Targets disease
index as two or more distinct ontology terms — usually a MONDO or EFO disease
term and an HP phenotype term — and different GWAS get mapped to different
members of the pair. The pipeline then treats them as two unrelated diseases and
their evidence is split.

This is an **upstream mapping problem**. It is documented in
[opentargets/issues#4446](https://github.com/opentargets/issues/issues/4446),
where Open Targets scoped a deterministic merge rule, measured ~1,727 duplicate
groups on release 26.06, and agreed a curation plan with the EFO team. Nothing
here is a defect of this analysis.

Problem 2 — the disease → therapeutic-area mapping — is separate and lives in
`../ta-distribution/MAPPING_REVIEW.md`. The punchline of this folder is that
**the two problems are largely the same problem**.

## The issue's own example is live in our data

| study                        | trait                                               | mapped to     | therapeutic area |
| ---------------------------- | --------------------------------------------------- | ------------- | ---------------- |
| `GCST90476238`               | Degeneration of intervertebral disc (PheCode 722.6) | `HP_0008419`  | **other**        |
| 8 other GWAS Catalog studies | lumbar disc degeneration                            | `EFO_0004994` | musculoskeletal  |

One condition, two terms, 8 studies against 7, and one side falls into the
residual bucket.

## Method

The deterministic rule from issue #4446, reimplemented on **release 25.06** (no
ontology version change). Two terms are nominated as equivalent iff, across
_different_ ontologies and never in an ancestor/descendant relation:

- one term's `dbXRefs` contains the other's ontology id, **or**
- they share a fine-grained xref
  (OMIM/UMLS/SNOMED/MeSH/MedGen/DOID/NCIT/Orphanet/…) **and** an exact label or
  synonym string.

`PMID` and `uniprot` excluded; coarse `ICD`/`MedDRA` excluded; components
bridging two terms of the same ontology dropped.

**Sanity check against the Open Targets pilot** — 1,760 groups here on 25.06
against 1,727 on 26.06, with MONDO–Orphanet 1,436 vs 1,461 and HP–MONDO 147
vs 144. EFO–HP is higher here (81 vs 8) because several of those were already
coalesced by the identical-label rule before 26.06 was cut.

## Result: real, but small — and it is not an independent problem

### 1. Only 40 duplicate groups are _live_

A duplicate costs us nothing unless **two or more of its members are actually
used** by our studies.

| universe                 | size  | terms in any group | **live groups** | terms involved | lost if merged |
| ------------------------ | ----- | ------------------ | --------------- | -------------- | -------------- |
| qualifying disease terms | 2,320 | 194                | **40**          | 80             | 40 (**1.7%**)  |
| gPS disease list         | 1,394 | 127                | **16**          | 32             | 16 (**1.1%**)  |

The other ~100 groups that touch our data contribute one member each and are
harmless.

### 2. Merging them barely moves gPS or gps_TA

| metric                            | genes changed          | mean before → after        |
| --------------------------------- | ---------------------- | -------------------------- |
| gPS (unique diseases)             | 166 / 8,285 (**2.0%**) | 4.449 → 4.427 (**−0.48%**) |
| gps_TA (unique therapeutic areas) | 167 / 8,285 (**2.0%**) | 2.423 → 2.415              |

Largest single-gene drop: 3 diseases, 1 therapeutic area. Maxima unchanged (148
and 20).

### 3. The OR = 10.3 band moves by 1.6%

| band            | before | after | leaving | entering |
| --------------- | ------ | ----- | ------- | -------- |
| gps_TA ∈ [2, 5] | 4,019  | 3,981 | 52      | 14       |
| gPS ∈ [2, 5]    | 3,504  | 3,504 | 8       | 8        |

**66 genes (1.6% of the band) are reclassified**, in both directions. This is
the number to quote if a referee asks. It is not nothing, but the merge would
have to be wrong in a very specific direction to overturn a 10-fold enrichment.

### 4. …and it is the same problem as the `other` bucket

This is the finding worth carrying into the response letter.

- **15 of the 16 live gPS groups straddle a therapeutic-area boundary; 13 of
  them have `other` on one side.**
- **32 of the 40 collapsed qualifying terms come out of `other`** (574 → 542).
  Every other area moves by 0 to 2 terms.

The mechanism is mechanical: the HP twin descends from `phenotype`, so it maps
to no therapeutic-area root and lands in `other`; the MONDO/EFO twin maps to a
real area; and `other` is counted as the 23rd therapeutic area in
`uniqueTherapeuticAreas`. **A duplicated disease therefore inflates gps_TA by
exactly one, through the residual bucket.**

So excluding `other` from gps_TA — already the recommended fix for problem 2 —
neutralises most of this problem without any remapping at all.

### 5. The split is inside the GWAS Catalog, not FinnGen vs GWAS Catalog

The mechanism suggested in the issue thread (FinnGen → MONDO, GWAS Catalog → HP)
is not the dominant pattern in our data. The recurring pattern is
**PheCode-derived studies getting HP terms while curated studies of the same
condition get EFO or MONDO terms**, both sides `GCST`: peptic ulcer 6/25, disc
degeneration 8/7, movement disorder 3/12 — all GCST on both sides.

## Two specific worries, both negative

### A. Could a remap deflate the non-European ancestry gains? No — structurally impossible

Novelty in `../ancestry-vs-sample-size/01_discovery_regression.ipynb` is defined
**per gene**:

```python
first_date = rows.groupby("geneId")["publicationDate"].transform("min")
rows["is_novel"] = rows["publicationDate"].eq(first_date)
```

**`diseaseIds` appears nowhere in that notebook.** A gene first reported by any
study on any trait is not novel for anybody afterwards, whichever ontology term
either study was mapped to. Remapping cannot change a single novelty count, and
the IRR = 1.67 for non-EUR is untouched.

The one indirect route would be a merged term moving a study between the disease
and measurement domains, which are modelled separately. It cannot: **0 of the 32
terms in live duplicate groups is a measurement term** (asserted in the
notebook).

And even if novelty were redefined per gene–disease pair — which a referee could
ask for — the mechanism still would not fire, because **ancestry classes are
mixed on both sides of essentially every twin**
(`dupterms_ancestry_split-r1.csv`). There is no group where one term collects
the European studies and the other the non-European ones:

| group                               | side 1                    | side 2                    |
| ----------------------------------- | ------------------------- | ------------------------- |
| uterine fibroid / Uterine leiomyoma | EUR=7, mixed=2, non-EUR=8 | EUR=5, mixed=1            |
| Peptic ulcer / peptic ulcer disease | EUR=1, mixed=1, non-EUR=1 | EUR=7, mixed=3, non-EUR=1 |
| anemia (phenotype) / anemia         | EUR=2, mixed=3, non-EUR=2 | EUR=2, mixed=1, non-EUR=3 |
| Dementia / dementia                 | non-EUR=5                 | EUR=6, mixed=2, non-EUR=3 |

The two most lopsided cases — Dementia and spondyloarthropathy/spondylosis — are
a handful of studies, and the second is not a true duplicate anyway.

### B. Could a remap increase the ChEMBL overlap? Yes in principle, by one pair in practice

The concern is well founded a priori. The enrichment propagates L2G evidence
**up the disease ontology** (term _D_ counts for _D_ and all its ancestors, max
score) and joins against ChEMBL indications, which are **not** propagated. An HP
term's ancestors are all HP/phenotype terms, never MONDO — so a GWAS on the HP
twin cannot match a ChEMBL indication in the MONDO tree.

The propagation was reimplemented in pandas and **reproduces the published
parquet exactly** (151,704 pairs for all evidence, 22,509 for PAV, identical
sets) before anything was changed. The merge uses **union semantics** — a
coalesced node inherits both ancestor closures, which is what OT's coalescing
means — not replacement, which would spuriously _lose_ pairs.

| definition   | ChEMBL pairs | supported before | supported after | **gained**       | lost |
| ------------ | ------------ | ---------------- | --------------- | ---------------- | ---- |
| all evidence | 37,377       | 742              | 743             | **1** (approved) | 0    |
| PAV only     | 37,377       | 161              | 161             | **0**            | 0    |

The single recovered pair is `ENSG00000110148` (CCKBR) × `HP_0004398` peptic
ulcer, an approved indication.

Why so little: ChEMBL's indications already include 254 HP terms, and our GWAS
terms already propagate to their full ancestor closure. For a remap to gain a
pair, a ChEMBL indication has to sit _exactly_ on the orphaned twin's side of
the split.

### What this does to OR = 10.3

| scenario                         | genes in window | supported pairs | approved | **OR**     | RS    | P           |
| -------------------------------- | --------------- | --------------- | -------- | ---------- | ----- | ----------- |
| published (no remap)             | 4,028           | 87              | 51       | **10.289** | 4.844 | 8.1 × 10⁻²⁵ |
| remap, published TA window       | 4,028           | 87              | 51       | **10.289** | 4.844 | 8.1 × 10⁻²⁵ |
| remap **+ TA window recomputed** | 3,981           | 93              | 52       | **9.212**  | 4.620 | 6.1 × 10⁻²⁴ |

The published OR reproduces exactly, which validates the whole chain. The remap
alone changes nothing. The only movement — **10.3 → 9.2** — comes from genes
crossing the 2–5 therapeutic-area boundary when gps_TA is recomputed after the
merge, i.e. from **problem 2, not problem 1**, and it leaves the conclusion
intact.

## Precision: not every nomination should be merged

Open Targets' LLM benchmark on the same rule scored ~86% equivalent, ~13%
broader/narrower, ~0.5% wrong, with HP–MONDO worst at ~38% broader/narrower.
Manual read of all 16 live gPS groups (this notebook's judgement, not a
benchmark):

| verdict          | n      | examples                                                                                                                                                                            |
| ---------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| equivalent       | **10** | anemia, uterine fibroid = uterine leiomyoma, peptic ulcer, dementia, cleft palate, disc degeneration, glomerulonephritis, hyperlipidemia, sensorineural hearing loss, iridocyclitis |
| broader/narrower | 3      | joint disease ↔ arthropathy; kidney disease ↔ nephropathy; movement disorder ↔ abnormality of movement                                                                           |
| **different**    | 2      | spondyloarthropathy (inflammatory) ↔ spondylosis (degenerative); frozen shoulder (adhesive capsulitis) ↔ bursitis                                                                 |
| **do not merge** | 1      | auditory system disease ↔ `MONDO_0021205`, which **is the `disorder of ear` therapeutic-area root** — merging it would break the hierarchy                                         |

Merging only the 10 safe groups would roughly halve every impact number above.
**The figures reported here are therefore an upper bound on the true
correction.**

## Range, not a point estimate

| rule                                    | live groups | terms affected | % of 2,320 |
| --------------------------------------- | ----------- | -------------- | ---------- |
| deterministic (xref / xref + synonym)   | 40          | 80             | 3.45       |
| loose (shared exact label/synonym only) | 64          | 125            | 5.39       |

The loose rule is an upper bound in the literal sense only — it pairs
_gonorrhea_ with _gastric cancer_ (shared "GC" synonym), _epilepsy_ with
_Seizure_, _neurodegenerative disease_ with _Brain atrophy_. Quote 40, not 64.

## Verdict

**Remapping is not on the critical path for this revision.** The maximum effect
of merging every nominated duplicate is a 0.5% shift in mean gPS and a 1.6%
reclassification of the OR = 10.3 band; the realistic effect, restricted to the
10 defensible merges, is about half that.

What _is_ worth acting on is the interaction. Fix `other` first — exclude it
from gps_TA and shrink it with the HP organ-system crosswalk described in
`../ta-distribution/MAPPING_REVIEW.md` — then re-measure. Most of the
duplication damage disappears as a side effect, because it was never really
about duplication.

For the response letter: the honest statement is that duplicated ontology terms
are a known upstream issue (#4446, with a curation plan agreed with EFO), that
they affect **1.7% of our disease terms and 2.0% of our genes**, that the effect
on gPS is **−0.5%**, and that the analysis is robust to it.

## Limitations

- Release **25.06 only**, deliberately — no ontology version change was made.
- Cross-ontology only: same-ontology near-duplicates are invisible to this rule,
  as are pairs with no shared xref and no shared exact synonym. 40 is a floor.
- The merge is applied at term level for counting. A real remap would also have
  to reconcile study-level trait names, the credible-set tables and the ChEMBL
  indication join.
- The precision verdicts are a manual reading of 16 groups, not an independent
  benchmark.

## Notebook

| Notebook                         | Needs Spark | Runtime | What it does                                                                                                                          |
| -------------------------------- | ----------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `01_duplicate_term_impact.ipynb` | no          | ~5 min  | rule, validation vs #4446, live groups, gPS/gps_TA/band impact, bounds, triage, ancestry-novelty check, ChEMBL overlap + OR scenarios |

```bash
cd chapters/06-review-r1/ontology-duplicates
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 01_duplicate_term_impact.ipynb
```

## Exports (all in `data/intermediate_files/`)

| File                                  | Contents                                                                  |
| ------------------------------------- | ------------------------------------------------------------------------- |
| `dupterms_live_summary-r1.csv`        | live duplicate groups per trait universe                                  |
| `dupterms_live_groups-r1.csv`         | every live gPS group: terms, study split, projects, areas, manual verdict |
| `dupterms_metric_impact-r1.csv`       | gPS and gps_TA before/after merging                                       |
| `dupterms_band_impact-r1.csv`         | movement across gps_TA ∈ [2,5]                                            |
| `dupterms_distribution_effect-r1.csv` | effect on the R3-mn-4 therapeutic-area distribution                       |
| `dupterms_bounds-r1.csv`              | deterministic versus loose rule                                           |
| `dupterms_verdict-r1.csv`             | the five questions and their answers                                      |
| `dupterms_ancestry_split-r1.csv`      | ancestry class of the studies on each side of every live twin             |
| `dupterms_chembl_overlap-r1.csv`      | ChEMBL T–I pairs supported before and after the remap                     |
| `dupterms_or_scenarios-r1.csv`        | OR = 10.3 under no remap, remap, and remap + recomputed window            |

---

# Notebook 02 — every GWAS mapped to an HP term, with a suggested disease term

`02_hp_mapped_gwas_review.ipynb` →
`data/intermediate_files/hp_mapped_gwas_review-r1.csv`

A **curation worklist**, not an applied change. Nothing downstream is modified.

## Scope and shape

The 1,394-term disease list behind gPS (diseases with ≥1 qualifying credible set
carrying an L2G-prioritised gene). Within it **212 terms are HP**, used by **545
studies** → **560 rows**, one per study × HP term.

Columns: `studyId`, `projectId`, `traitFromSource`, `all_diseaseIds`,
`n_diseaseIds`, `hp_term`, `hp_name`, `hp_description`,
`hp_current_therapeutic_area`, `suggested_term`, `suggested_name`,
`suggested_therapeutic_area`, `suggested_term_already_in_disease_list`,
`confidence`, `reason`.

Current areas of those 560 rows: **512 `other`**, 41 `sign or symptom`, 6
cardiovascular, 1 visual — i.e. 99% carry no real therapeutic area, which is the
whole point.

## How the suggestion is made — and why no id is invented

Two stages, deliberately separated:

1. **Retrieval — deterministic, from `disease.parquet` only.** Per HP term, up
   to 12 candidates from direct cross-reference, shared fine-grained
   cross-reference, and label/synonym token overlap, restricted to non-HP terms
   that map to a _real_ therapeutic area (`other` and `sign or symptom` excluded
   as targets).
2. **Ranking — eight Haiku agents**, one per chunk of ~27 terms, each allowed to
   return **only an id present in that term's own candidate list**, or decline.

Both the shortlist (`hp_review_candidates.json`) and the model output
(`hp_review_verdicts.json`) are frozen in this folder, so the table rebuilds
without re-running any model.

**Validation result: 96 suggestions, 96 pass every check** — id exists in
`disease.parquet`, id was in that term's candidate list, id is not itself HP, id
maps to a real therapeutic area. **0 rejected.** No ontology identifier in the
table comes from model memory.

One systematic correction is applied in the notebook: `0` means _the GWAS trait
does not match its HP term_, but two verdicts used `0` to mean _the candidate
list was useless_, which is `NA`. Any `0` whose reason talks about candidates is
reclassified; 2 of 3 were.

## Result

| confidence | meaning                       | HP terms | GWAS rows | studies |
| ---------- | ----------------------------- | -------- | --------- | ------- |
| **1**      | confident match               | **70**   | 232       | 231     |
| **2**      | needs checking                | 26       | 60        | 60      |
| **NA**     | no disease term exists        | **115**  | 267       | 256     |
| **0**      | HP mapping itself looks wrong | 1        | 1         | 1       |

Of the 70 confident matches: **31 targets are already in the 1,394-disease
list** (remapping removes a duplicate term) and **39 are new to it** (pure
relabel — the study keeps its identity but gains a real therapeutic area). They
would move into 16 areas, led by cardiovascular (10), nervous (8),
gastrointestinal (8) and visual (8).

Examples: Abnormal bleeding → hemorrhagic disease; Abnormal cardiac septum
morphology → heart septal defect; Acute kidney injury → acute kidney failure;
Cleft palate → cleft palate; Bronchiectasis → bronchiectasis; Cerebral ischemia
→ brain ischemia; Cholecystitis → Cholecystitis, Acute; Blindness → blindness
(disorder).

The single confidence-0 flag: `FINNGEN_R12_N14_MESNRUIRREG`, trait _"Excessive,
frequent and irregular menstruation"_, mapped to `HP_0000876` **Oligomenorrhea**
— which means _infrequent_ menses. Upstream curation bug, worth reporting rather
than remapping.

## The negative result is the important one

**115 of 212 HP terms (54%) have no disease equivalent**, covering 267 of the
560 rows: Inguinal hernia (15 studies), Iron deficiency anemia (13), Back pain
(10), Hallux valgus, Proteinuria, Syncope, Thrombophlebitis, Umbilical hernia,
Menorrhagia, Urinary incontinence, Sepsis, Tinnitus, Cough. EFO and MONDO model
these as phenotypes by design; there is nothing to remap them onto.

**So remapping cannot close the coverage gap — at best it fixes 96 of 212
terms.** The remaining 115 can only be handled by an HP organ-system →
therapeutic-area crosswalk (`../ta-distribution/MAPPING_REVIEW.md`), which is
why that remains the higher-value fix.

## Limitations

- A model's choice from a deterministic shortlist. Confidence-1 rows should
  still be eyeballed before use.
- Retrieval can only propose what token overlap or a cross-reference surfaces; a
  correct target with an unrelated label and no shared xref is missed and
  appears as NA.
- Frozen model output — re-running the agents would not reproduce it verbatim.
  The validation cell, not the model, is what guarantees the ids are real.

## Run

```bash
cd chapters/06-review-r1/ontology-duplicates
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 02_hp_mapped_gwas_review.ipynb
```

| Export                                 | Contents                                                                                                  |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **`hp_terms_review-r1.tsv`**           | **one row per HP term (212 = 70 + 26 + 115 + 1)** with all its GWAS in one column — the sheet to validate |
| **`hp_mapped_gwas_review-r1.tsv`**     | all 560 GWAS × HP rows, same verdicts, one study per row                                                  |
| `hp_mapped_gwas_review-r1.csv`         | the 560-row worklist, comma-separated                                                                     |
| `hp_mapped_gwas_review_summary-r1.csv` | HP terms / studies / rows by confidence                                                                   |

Both TSVs carry `candidates_offered` (the shortlist the model was restricted to,
so a reviewer can see what else was available and catch a better option that was
passed over) plus blank `validated_ok` and `reviewer_note` columns to fill in.
Every field is stripped of tabs and newlines, and the notebook asserts both
files have perfectly rectangular rows.

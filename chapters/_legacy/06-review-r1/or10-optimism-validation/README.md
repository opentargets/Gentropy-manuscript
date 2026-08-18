# R2-MJ-2 — optimism bias in the OR = 10.3 claim

Round-1 referee 2, major comment 2, verbatim:

> I'm suspicious that the post-hoc selection of variables to predict TI-pair
> success (those with PAVs and genetic support in 2-5 TAs) achieving an OR of 10
> is valid. This is not replicated from a model in a hold-out group, but appears
> to be what the authors have selected to maximize the OR. This is almost
> certainly a case of optimism bias, where the authors have picked thresholds
> that maximize the result. I would be convinced this were not the case if the
> authors trained a model in separate data and tested in a held-out dataset.
> This is does not seem to have been the case.

He is right that no held-out validation was performed. The task here is to
quantify the optimism and report a defensible estimate, not to argue the charge
away.

## Notebooks

Run in order; each depends on the previous one's exports.

| Notebook                         | Needs Spark       | Runtime | What it does                                                                                 |
| -------------------------------- | ----------------- | ------- | -------------------------------------------------------------------------------------------- |
| `01_build_pair_tables.ipynb`     | yes, 40 GB driver | ~4 min  | Builds the pair-level master tables and reproduces every published number from them          |
| `02_phase0_threshold_grid.ipynb` | no                | seconds | Phase 0 — the full OR surface over therapeutic-area windows × PAV                            |
| `03_phase1_heldout_splits.ipynb` | no                | ~3 min  | Phase 1 — 200 held-out target splits, optimism, bias-corrected estimate                      |
| `04_phase2_pharmaprojects.ipynb` | no                | ~2 min  | Phase 2 — Pharmaprojects check, with a power calculation stated before the test              |
| `05_pleiotropy_ceiling.ipynb`    | no                | ~1 min  | Does the _ceiling_ (exclude highly pleiotropic targets) replicate on Pharmaprojects?         |
| `06_interaction_test.ipynb`      | no                | ~3 min  | The PAV × pleiotropy interaction, fitted explicitly, with a permutation P value per resource |

`or10_stats.py` holds `or_rs` (Fisher OR, relative success, Woolf CIs — a
reimplementation of
`gentropy.method.drug_enrichment_from_evid.chemblDrugEnrichment.drug_enrichemnt_from_evidence`)
and `support_mask` (one genetic-support definition). Notebooks 02–04 import it
so no definition can drift between phases; notebook 01 keeps its own inline copy
as the reproduction proof.

```bash
cd chapters/06-review-r1/or10-optimism-validation
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 \
  01_build_pair_tables.ipynb
# then 02, 03, 04 the same way
```

## Method

### The pair-level reduction

`drug_enrichemnt_from_evidence` ends in a right join of the propagated
association table onto the ChEMBL target–indication pairs, followed by a Fisher
test. So the entire published enrichment is a function of one table with 37,377
rows: per T–I pair, the maximum L2G score propagated through the disease
ontology (all credible sets, and PAV-containing credible sets separately) plus
the gene-level therapeutic-area and disease counts. Notebook 01 builds that
table once in Spark; Phases 0–2 are pure pandas, so 200 splits and 1,000 power
simulations cost minutes rather than hours.

The therapeutic-area window is a gene-level filter and ontology propagation is
per target, so restricting the gene set before propagation is equivalent to
filtering the joined table afterwards. That equivalence is not assumed —
notebook 01 reproduces the published 2×2 table exactly from the
filtered-afterwards version.

### Splitting

By **target**, never by T–I pair. PAV status, gPS and therapeutic-area count are
gene properties, so a pair-level split would place the selected-on quantity in
both halves. Targets are stratified by their approved-pair count and alternately
assigned, which balances the scarce quantity (~25 approved strict-definition
pairs per half). 200 splits, not one: with 51 approved supported pairs a single
split is a coin flip.

### The search space is a modelling choice

Optimism belongs to the procedure being simulated. An unconstrained search over
all 51 windows — including single-value windows resting on five approved pairs —
models a worse procedure than the one that produced the paper, and yields more
optimism. Both are reported:

- `naive` — any window with ≥ 5 approved supported pairs in half A. The
  referee's literal charge, an upper bound.
- `reportable` — width ≥ 3 and ≥ 20 approved supported pairs in half A. What
  anyone would actually publish.

Three intermediate constraint levels are reported too, so the reader sees
optimism scale with search freedom rather than one number chosen by us.

## Results

### Reproduction (notebook 01)

Every published input number reproduces exactly from the pair-level table:
37,377 T–I pairs (phase I 6,163, II 14,410, III 12,240, IV 4,564), 2,734
strict-definition gene–disease associations, all-GWAS OR 3.618578 with 242
approved supported pairs, strict-definition OR 10.288962 and RS 4.843708 with
the 2×2 table [[32777, 4513], [36, 51]].

**The strict definition covers 51 of 242 approved GWAS-supported pairs (21.1%),
not 52 (21.5%).** The analysis brief and the manuscript say 52. The published
enrichment table itself reports 51, and 51 is the value consistent with every
other published number: 32777 × 51 / (4513 × 36) = 10.2890 (52 would give
10.4907), and the four cells sum to the published 37,377 pairs. Not introduced
by anything here — flagged for the response letter.

### Phase 0 — the window is a plateau, not a spike

Among the 64 PAV windows carrying at least 10 approved supported pairs, 2–5
ranks **4th**, and the three above it are ties within noise (TA = 2 alone, OR
18.0 on 10 approved pairs; 2–3, OR 10.8 on 18; 2–4, OR 10.35 on 40). Moving
either bound by one gives 8.46–10.35, a spread of 21% of the median. 32 of the
64 windows fall inside the published 95% CI [6.71, 15.78] and 45 of 64 have a CI
excluding the all-GWAS baseline of 3.62. Only 3 of 64 exceed the published point
estimate.

The window is also not where most of the effect comes from:

| Definition               | OR    | RS   | approved supported pairs |
| ------------------------ | ----- | ---- | ------------------------ |
| all GWAS support         | 3.62  | 2.76 | 242                      |
| PAV, no window           | 5.89  | 3.71 | 72                       |
| any support + 2–5 TA     | 3.95  | 2.92 | 139                      |
| PAV + 2–5 TA (published) | 10.29 | 4.84 | 51                       |

The window alone adds almost nothing (3.62 → 3.95); PAV alone gives 5.89. The
published value needs both, and the interaction is the claim.

The odds ratio by exact therapeutic-area count shows the peak the window
brackets: 1.2 (TA = 1), 18.0, 7.2, 9.9, 9.9 (TA = 2–5), then 5.1, 3.2, 2.9 (TA =
6–8) and 1.7 at TA = 12. The gPS surface behaves the same way — PAV + gPS 2–5
gives OR 11.4.

### Phase 1 — the frozen definition holds out; a naive search would have inflated it

**The published definition, held out (no selection anywhere).** Median OR across
200 splits **10.32**, split-to-split spread 6.62–16.75, optimism factor **0.97**
[Monte Carlo 0.91–1.04] — no optimism at all, which is expected for a definition
that is not being re-chosen on each half. 95.5% of splits have their own 95% CI
lower bound above 3.62. This answers "does the published definition survive out
of sample", and it does.

**Selection optimism**, i.e. what a threshold search would have cost:

| Search space                        | 2–5 selected | in-sample OR | held-out OR | optimism factor       | corrected OR |
| ----------------------------------- | ------------ | ------------ | ----------- | --------------------- | ------------ |
| naive (≥5 approved, any width)      | 0%           | 22.28        | 9.17        | **2.63** [2.32, 2.99] | 3.91         |
| ≥10 approved, any width             | 7.5%         | 13.93        | 7.89        | 1.78 [1.62, 1.96]     | 5.77         |
| ≥20 approved, any width             | 31.3%        | 11.00        | 8.88        | 1.24 [1.15, 1.33]     | 8.33         |
| reportable (≥20 approved, width ≥3) | 34.8%        | 10.86        | 9.04        | **1.19** [1.11, 1.28] | **8.63**     |
| ≥30 approved, width ≥3              | 24.9%        | 9.15         | 8.15        | 1.11 [1.03, 1.18]     | 9.32         |

Under the `reportable` search space the bias-corrected estimate is **OR 8.63**
(published CI scaled by the same factor: 5.63–13.24) and **RS 4.50**. Under the
`naive` search space it is OR 3.91 — essentially the all-GWAS baseline — but
that search space selects single-value windows sitting on a median of 7 approved
pairs, which is not how the definition was derived and not something anyone
would report.

The published window is the single most frequently selected one whenever the
search is constrained to reportable windows (34.8%, next is 2–4 at 31%), but it
is _not_ selected in a majority, so the brief's secondary success criterion is
**not met**.

**Premise stability** (half-samples, labelled as stability, not independence):
PAV enrichment points the same way in 99.5% of halves and reaches p < 0.05 in
79.5%; the non-linearity LRT reaches p < 0.05 in 100% of halves and p < 1e-4 in
92.5%. On the full data these reproduce the published values — PAV OR 6.05
versus 3.09, p = 2.4e-4; quadratic LR = 64.897, p = 7.9e-16, fitted peak at 1.90
therapeutic areas.

**Success criterion.** Held-out interval excludes the all-GWAS baseline 3.62 for
the frozen window (2.5th percentile 6.62) and for every search space constrained
to ≥20 approved pairs (4.96 for `reportable`). It does **not** for the naive
search (1.47).

### Phase 2 — Pharmaprojects

Self-validation first: regressing launch on Pharmaprojects' own genetic-support
flag gives OR 2.32 [1.94, 2.78], p = 1.8e-20 — the ~2× Nelson 2015 and Minikel
2024 report, so the table is joined correctly. 7,390 T–I pairs, 913 launched.

**Overlap.** 447 of 911 launched T–I pairs are also ChEMBL phase 4 (49.1%). This
is independent curation of the same pharmacological reality, not independent
data, and must not be called replication.

| Test                                           | Result                                                                                                                                                                        |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PAV versus non-PAV (never tested here before)  | OR **2.34** versus **1.40**, difference p = **0.040**. Same direction as ChEMBL (6.05 versus 3.09, p = 2.4e-4), nominally significant. 33 launched of 137 PAV-supported pairs |
| Frozen PAV + 2–5 TA definition                 | OR **4.50** [2.66, 7.60], RS 3.16, p = 2.5e-7, on 23 launched of 60 supported pairs                                                                                           |
| Same, relative to that resource's own baseline | all-GWAS support gives OR 1.65 there, so the strict definition is a **2.72×** lift — against **2.84×** in ChEMBL                                                              |
| Non-linear pleiotropy                          | **not detected**, p = 0.49                                                                                                                                                    |

Every effect is weaker in Pharmaprojects (all-GWAS OR 1.65 versus 3.62), so
power was computed under two effect sizes: ChEMBL slopes unchanged (99% power)
and slopes scaled by the observed attenuation factor 0.392 on the log-odds scale
(**43%** power). The fair figure is 43%, so the null on non-linearity is a
**pre-stated power limitation, not evidence against it**. Worth stating
precisely: the Pharmaprojects quadratic coefficient is −0.042 [−0.164, 0.080],
whose interval excludes the ChEMBL point estimate of −0.267 — so a
ChEMBL-_magnitude_ non-linearity is excluded there, while an attenuated one is
not.

### The pleiotropy ceiling replicates on Pharmaprojects (notebook 05)

Phase 2 tested the 2–5 window as one block, which conflates two claims. The
floor (TA ≥ 2) only removes targets with very little genetic evidence; the
**ceiling (TA ≤ 5) is the substantive claim** — broad pleiotropy predicts
failure. Tested on its own, with no-support pairs as the common reference and
supported pairs at TA ≤ 1 excluded so the contrast is purely the ceiling:

| Resource       | support | low (TA 2–5)      | high (TA ≥ 6)      | low / high | p          |
| -------------- | ------- | ----------------- | ------------------ | ---------- | ---------- |
| ChEMBL         | PAV     | OR 10.32 (51/87)  | OR 3.10 (20/67)    | 3.33       | 0.00048    |
| Pharmaprojects | PAV     | OR 4.50 (23/60)   | **OR 1.09** (9/69) | **4.14**   | **0.0014** |
| ChEMBL         | any     | OR 4.01 (139/398) | OR 2.97 (81/285)   | 1.35       | 0.073      |
| Pharmaprojects | any     | OR 1.88 (41/202)  | OR 1.48 (39/233)   | 1.27       | 0.34       |

So the ceiling replicates, and by a larger factor than in ChEMBL. Among
PAV-supported pairs in Pharmaprojects, a highly pleiotropic target has **OR
1.09** — genetic support confers no advantage at all once the target is broadly
pleiotropic.

Not a threshold artefact: sweeping the cut point from 3 to 10, the low/high
ratio exceeds 1 at 8 of 8 cut points in Pharmaprojects PAV support and is
significant at 7 of 8. The bin profile shows the same peak-then-decline shape as
ChEMBL — Pharmaprojects PAV support gives 1.04 (TA 1), 4.14 (2–3), 4.72 (4–5),
1.36 (6–9), 0.78 (10+); ChEMBL gives 1.21, 10.93, 10.02, 3.50, 2.65.

The same question in gPS, all four contrasts with the P value for the difference
(`or10_ceiling_gps_contrast-r1.csv`):

| Resource       | support | gPS ≤ 5          | gPS ≥ 10          | ratio | p          |
| -------------- | ------- | ---------------- | ----------------- | ----- | ---------- |
| ChEMBL         | PAV     | OR 9.11 (20/36)  | OR 4.69 (38/97)   | 1.94  | 0.093      |
| ChEMBL         | any     | OR 4.80 (86/220) | OR 2.97 (104/366) | 1.62  | **0.0077** |
| Pharmaprojects | PAV     | OR 3.79 (11/32)  | OR 1.43 (14/85)   | 2.66  | **0.039**  |
| Pharmaprojects | any     | OR 1.92 (25/121) | OR 1.62 (48/266)  | 1.18  | 0.54       |

The ChEMBL any-support row is the published split (OR 4.8 versus 3.0, P = 0.008)
and reproduces exactly. Note the pattern across the four rows: the ratio is
larger under PAV support in both resources but the P value is worse in ChEMBL
PAV (0.093) purely because that stratum has 36 versus 220 pairs. The
Pharmaprojects any-support contrast is null (p = 0.54) — with 25 and 48
launches, that is a weak test, not evidence against the ceiling.

**Two consequences.** First, the ceiling only bites among PAV-supported pairs —
without PAV it is a weak, non-significant trend in both resources (1.35 and
1.27). The published combination of PAV with intermediate pleiotropy is
therefore an interaction, not two additive filters, and that is what transfers.
Second, this reconciles the Phase 2 non-linearity null: that test used
continuous `log(TA + 1)` terms on _all_ supported pairs, which is exactly the
stratum where the ceiling is weakest, and it was underpowered on top of that.

### The interaction itself, with a P value (notebook 06)

Stratum-specific P values are not a test of an interaction, so it is fitted
explicitly: `outcome ~ pav + low + pav:low` on **supported pairs only** (`low` =
TA 2–5 versus ≥ 6; supported pairs at TA ≤ 1 excluded, so `low` is
intermediate-versus-high pleiotropy and not the floor).

Success rates in the four cells make the interaction visible before any model:

|                        | no PAV          | PAV                |
| ---------------------- | --------------- | ------------------ |
| ChEMBL, TA ≥ 6         | 27.98% (61/218) | 29.85% (20/67)     |
| ChEMBL, TA 2–5         | 28.30% (88/311) | **58.62% (51/87)** |
| Pharmaprojects, TA ≥ 6 | 18.29% (30/164) | 13.04% (9/69)      |
| Pharmaprojects, TA 2–5 | 12.68% (18/142) | **38.33% (23/60)** |

Without a PAV, the pleiotropy ceiling does nothing in either resource (28.0%
versus 28.3% in ChEMBL; if anything reversed in Pharmaprojects). With a PAV it
is large in both.

| Resource       | interaction OR | 95% CI     | Wald    | LRT     | permutation (10,000) |
| -------------- | -------------- | ---------- | ------- | ------- | -------------------- |
| ChEMBL         | 3.28           | 1.51–7.13  | 0.0028  | 0.0024  | **0.0018**           |
| Pharmaprojects | 6.39           | 2.17–18.79 | 0.00075 | 0.00050 | **0.0010**           |

`low` is permuted within PAV strata, so the null preserves both main effects and
all four cell sizes; that P value is the one to quote given 9–23 successes per
cell. **The Pharmaprojects value is the replication test: p = 0.0010 two-sided,
0.0007 one-sided** with the direction fixed in advance by ChEMBL. The three-way
`pav:low:resource` term is not significant (OR 1.95, p = 0.32), i.e. the two
estimates are compatible — but that test is not valid as an independence check,
because half the successes are shared.

Two limits, both real:

- **Measure-specific in ChEMBL.** Sweeping the therapeutic-area cut point, the
  interaction holds at cuts 5–9 in ChEMBL (OR 2.2–3.3, p 0.042–0.0024) and at 7
  of 8 cut points in Pharmaprojects (OR 2.7–13.2). In **gPS** it is absent in
  ChEMBL at every cut (OR 0.62–1.25, p ≥ 0.44) while present in Pharmaprojects
  at cuts ≥ 8 (OR 4.1–7.1, p 0.006–0.0002). So the interaction is specific to
  _therapeutic-area breadth_ in ChEMBL, not to pleiotropy measured any way.
- **The non-overlapping subset is uninformative, not negative.** Removing every
  Pharmaprojects pair that also appears in ChEMBL leaves 236 supported pairs
  with 13 launches and 3 in the critical cell: interaction OR 1.64 [0.17,
  15.85], p = 0.67. Same direction, no power. This cannot be quoted either way.

## Inputs

Read by notebook 01, all under `data/`:

- `25.06/output/credible_set`, `output/study`, `output/disease/disease.parquet`,
  `output/evidence/sourceId=chembl`
- `intermediate_files/l2g_full_for_enrichment/` (70,400 rows) — the L2G table
  with `VEP`, `maf`, year
- `intermediate_files/genes_therapeutic_areas` (8,285 genes) —
  `uniqueTherapeuticAreas`, `uniqueDiseases`
- `intermediate_files/minikel_etal_processed_data_v2.csv` (7,390 rows) — the
  processed Pharmaprojects table from
  `chapters/05-other-drug-indication-data/01-process-minikel_etal_data.ipynb`
- `intermediate_files/df_for_enrichment_regression.csv` — used by notebook 02
  only, as an independent cross-check of the PAV column (it carries `max_vep`
  from a different aggregation; agreement is asserted pair by pair)

## Outputs

To `data/intermediate_files/`, all suffixed `-r1`:

| File                                                                                                                                                                                                                                  | Contents                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `ti_pairs_chembl_master-r1.parquet`                                                                                                                                                                                                   | the 37,377-row ChEMBL pair-level table                    |
| `ti_pairs_pharmaprojects_master-r1.parquet`                                                                                                                                                                                           | the 7,390-row Pharmaprojects equivalent                   |
| `l2g_indirect_assoc_all-r1.parquet`, `l2g_indirect_assoc_pav-r1.parquet`                                                                                                                                                              | propagated scores (151,704 / 22,509 target–disease pairs) |
| `chembl_ti_pairs_maxphase-r1.parquet`                                                                                                                                                                                                 | ChEMBL pairs with max clinical phase                      |
| `or10_phase0_grid_named-r1.csv`, `..._full-r1.csv`, `or10_phase0_ta_profile-r1.csv`, `..._gps-r1.csv`                                                                                                                                 | Phase 0 surfaces                                          |
| `or10_phase1_frozen_window_splits-r1.csv`, `..._search_splits-r1.csv`, `..._premises-r1.csv`, `..._optimism-r1.csv`, `..._criteria-r1.csv`, `..._window_frequency-r1.csv`, `..._premise_stability-r1.csv`, `..._split_balance-r1.csv` | Phase 1, per split and summarised                         |
| `or10_phase2_pav_strata-r1.csv`, `..._pav_contrast-r1.csv`, `..._frozen_definition-r1.csv`, `..._window_profile-r1.csv`, `..._nonlinearity-r1.csv`, `..._power-r1.csv`, `..._power_simulation-r1.csv`                                 | Phase 2                                                   |
| `or10_ceiling_strata-r1.csv`, `..._test-r1.csv`, `..._cut_sweep-r1.csv`, `..._bins-r1.csv`, `..._gps-r1.csv`                                                                                                                          | notebook 05, the pleiotropy ceiling                       |
| `or10_interaction_cells-r1.csv`, `..._test-r1.csv`, `..._sweep-r1.csv`, `..._non_overlap-r1.csv`                                                                                                                                      | notebook 06, the interaction                              |

Plus three reproduction records next to the notebooks:
`reproduction_checks-r1.csv`, `pharmaprojects_checks-r1.csv`,
`chembl_pharmaprojects_overlap-r1.csv`.

`response_log_snippet.md` in this folder is the paste-ready summary for the
manuscript repo's `R_202606/response_log.md`.

## Known issues and limitations, deliberately left as they are

- **No figures.** Numbers only at this stage, by instruction. The Phase 0
  heatmap input is `or10_phase0_grid_full-r1.csv` and the Phase 1 held-out
  distribution input is `or10_phase1_search_splits-r1.csv` when figures are
  wanted.
- **The `naive` versus `reportable` distinction is a judgement call.** The
  corrected OR is 3.91 or 8.63 depending on which search space is taken as the
  model of what happened. Both are reported with their assumptions; picking one
  for the response letter is an editorial decision, not a computational one.
- **Monte Carlo error, not a confidence interval.** The interval on every
  optimism factor is the standard error of the mean across 200 splits, which
  measures how well 200 splits pin the expectation down. It does not account for
  sampling of the underlying data. The much wider split-to-split spread is
  reported in the same table.
- **Corrected CIs are the published CI divided by the point-estimate factor.**
  They ignore uncertainty in the factor itself, so they are narrower than a
  fully propagated interval would be.
- **Held-out estimates on thin cells.** Under the naive search space, 9 of 200
  splits produce a held-out 2×2 table with an empty cell; the OR is not
  estimable there and those splits are dropped from the optimism average rather
  than being counted as OR = 1. The count is printed in the notebook for every
  search space, and `≥30 approved, width ≥3` fails to find any selectable window
  in 31 of 200 splits.
- **Pharmaprojects tests are underpowered by construction** and 49% of its
  launched pairs are shared with ChEMBL. Neither is fixable; both are stated
  wherever the numbers appear.
- **`gps` sensitivity in Phase 0 is secondary.** The claim under attack is
  stated in therapeutic areas; the gPS surface is reported because the
  manuscript also quotes a gPS split, not because it is being validated here.

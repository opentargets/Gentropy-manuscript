# R2-MJ-2 (optimism bias in OR = 10.3) — results summary

Analysis: `Gentropy-manuscript/chapters/06-review-r1/or10-optimism-validation/`,
four notebooks, all executed. Every published input number reproduces exactly
from a 37,377-row pair-level table.

## Accepted premise

No held-out validation was performed in the submitted manuscript. That is
correct and is conceded.

## 1. Threshold sensitivity — the window is a plateau

Of the 64 PAV × therapeutic-area windows carrying ≥ 10 approved supported pairs,
2–5 ranks 4th; the three above it are ties within noise (TA = 2 alone, OR 18.0
on 10 approved pairs; 2–3, OR 10.8 on 18; 2–4, OR 10.35 on 40). Moving either
bound by one gives OR 8.46–10.35 (21% of the median). 32 of 64 windows lie
inside the published 95% CI, and 45 of 64 have CIs excluding the all-GWAS
baseline of 3.62. Only 3 of 64 exceed the published point estimate, all on ≤ 18
approved pairs.

Decomposition: all-GWAS support OR 3.62 → PAV with no window 5.89 → any
support + 2–5 TA 3.95 → PAV + 2–5 TA 10.29. The window alone contributes almost
nothing; PAV alone contributes most of the lift; the published value requires
both.

## 2. Held-out validation (200 target-level splits within ChEMBL)

Split by target, never by target–indication pair, because PAV status and
pleiotropy are gene properties; targets stratified on approved-pair count (~25
approved strict-definition pairs per half).

- **The published definition, frozen and evaluated on untouched halves**: median
  OR **10.32** (split spread 6.62–16.75), optimism factor **0.97**. The
  definition does not degrade out of sample. 95.5% of splits have their own 95%
  CI lower bound above 3.62.
- **If a threshold search had been used**, optimism depends on how free that
  search is allowed to be. Selecting the OR-maximising window on one half and
  evaluating it on the other gives an optimism factor of **1.19** [MC 1.11–1.28]
  when the search is limited to windows anyone would publish (≥ 20 approved
  supported pairs, width ≥ 3), and **2.63** [2.32–2.99] when any window with ≥ 5
  approved pairs is allowed. The corresponding bias-corrected estimates are **OR
  8.63** (RS 4.50) and OR 3.91.
- **Window-selection frequency**: 2–5 is the most frequently selected window
  under the constrained search (34.8%, next 2–4 at 31%) but not a majority;
  under an unconstrained search it is never selected, because narrow windows on
  ~7 approved pairs win.
- **Premise stability** (half-samples; stability, not independence — both
  premises were established on this data): PAV enrichment same direction in
  99.5% of halves, p < 0.05 in 79.5%; non-linearity LRT p < 0.05 in 100% of
  halves.

Pre-specified primary criterion — the held-out interval excludes the all-GWAS
baseline 3.62 — is met for the frozen definition (2.5th percentile 6.62) and for
the constrained search (4.96). The secondary criterion, 2–5 selected in a
majority of splits, is not met.

## 3. Pharmaprojects

Independent curation of the same pharmacological reality, **not** independent
data: 447 of 911 launched target–indication pairs are also ChEMBL phase 4
(49.1%). Validated first against its own evidence — Pharmaprojects' own
genetic-support flag gives OR 2.32 [1.94, 2.78], the ~2× reported by Nelson 2015
and Minikel 2024.

- **PAV versus non-PAV (never tested there before)**: OR **2.34** versus
  **1.40**, difference p = **0.040** — same direction as ChEMBL (6.05 versus
  3.09, p = 2.4e-4).
- **Frozen PAV + 2–5 TA definition**: OR **4.50** [2.66, 7.60], RS 3.16, p =
  2.5e-7, on 23 launched of 60 supported pairs. Against that resource's own
  all-GWAS baseline of 1.65 this is a **2.72×** lift, against **2.84×** in
  ChEMBL — the relative enrichment transfers even though every absolute odds
  ratio is smaller there.
- **Non-linear pleiotropy**: not detected (p = 0.49). Power stated in advance:
  99% against a full-strength ChEMBL effect but **43%** once slopes are scaled
  by the attenuation actually observed in this resource (factor 0.392 on the
  log-odds scale). A pre-stated power limitation, not evidence against the
  non-linearity. The Pharmaprojects quadratic coefficient is −0.042 [−0.164,
  0.080], which excludes the ChEMBL estimate of −0.267, so a ChEMBL-magnitude
  non-linearity is excluded there.

## 4. The pleiotropy ceiling — the substantive part of the window — does replicate

The 2–5 window is two claims. The floor (TA ≥ 2) mostly removes targets with
little genetic evidence; the **ceiling (TA ≤ 5) is the claim of interest**:
highly pleiotropic targets should be excluded. Tested on its own, against
no-support pairs as the common reference:

| Resource       | support | low (TA 2–5)     | high (TA ≥ 6)      | low / high | p          |
| -------------- | ------- | ---------------- | ------------------ | ---------- | ---------- |
| ChEMBL         | PAV     | OR 10.32 (51/87) | OR 3.10 (20/67)    | 3.33       | 0.00048    |
| Pharmaprojects | PAV     | OR 4.50 (23/60)  | **OR 1.09** (9/69) | **4.14**   | **0.0014** |
| ChEMBL         | any     | OR 4.01          | OR 2.97            | 1.35       | 0.073      |
| Pharmaprojects | any     | OR 1.88          | OR 1.48            | 1.27       | 0.34       |

The ceiling replicates in Pharmaprojects by a larger factor than in ChEMBL, and
PAV-supported pairs on highly pleiotropic targets show **no enrichment at all
there (OR 1.09)**. Sweeping the cut point from 3 to 10 therapeutic areas, the
low/high ratio exceeds 1 at 8 of 8 cut points and is significant at 7, so this
is not an artefact of the number 6. Bin profile, Pharmaprojects PAV support:
1.04 (TA 1), 4.14 (2–3), 4.72 (4–5), 1.36 (6–9), 0.78 (10+) — the same
peak-then-decline as ChEMBL (1.21, 10.93, 10.02, 3.50, 2.65). Same result in
gPS, with the difference tested in all four strata: Pharmaprojects PAV support
gives gPS ≤ 5 OR 3.79 versus gPS ≥ 10 OR 1.43 (ratio 2.66, p = 0.039); ChEMBL
PAV 9.11 versus 4.69 (ratio 1.94, p = 0.093, on 36 versus 97 pairs); ChEMBL any
support 4.80 versus 2.97 (ratio 1.62, p = 0.0077 — the published split);
Pharmaprojects any support 1.92 versus 1.62 (ratio 1.18, p = 0.54, on 25 and 48
launches, so a weak test rather than a negative one).

## 5. The interaction, tested formally, replicates at P = 0.001

The ceiling only bites among PAV-supported pairs, so the published definition is
an **interaction**, not two independent filters. Fitted explicitly on supported
pairs (`outcome ~ pav + low + pav:low`, `low` = TA 2–5 versus ≥ 6, supported
pairs at TA ≤ 1 excluded):

|                        | no PAV          | PAV                |
| ---------------------- | --------------- | ------------------ |
| ChEMBL, TA ≥ 6         | 27.98% (61/218) | 29.85% (20/67)     |
| ChEMBL, TA 2–5         | 28.30% (88/311) | **58.62% (51/87)** |
| Pharmaprojects, TA ≥ 6 | 18.29% (30/164) | 13.04% (9/69)      |
| Pharmaprojects, TA 2–5 | 12.68% (18/142) | **38.33% (23/60)** |

| Resource           | interaction OR | 95% CI         | LRT P       | permutation P (10,000) |
| ------------------ | -------------- | -------------- | ----------- | ---------------------- |
| ChEMBL             | 3.28           | 1.51–7.13      | 0.0024      | 0.0018                 |
| **Pharmaprojects** | **6.39**       | **2.17–18.79** | **0.00050** | **0.0010**             |

`low` is permuted within PAV strata, preserving both main effects and all four
cell sizes, so the permutation P value is the one to quote at these cell counts.
**The replication P value is 0.0010 two-sided (0.0007 one-sided, direction
pre-specified by ChEMBL).** The three-way `pav:low:resource` term is not
significant (OR 1.95, p = 0.32): the two estimates are compatible — though that
is not an independence test, since half the successes are shared.

Without a PAV the ceiling does nothing in either resource (28.0% versus 28.3% in
ChEMBL; slightly reversed in Pharmaprojects). That is the cleanest form of the
claim to put in the response: _protein- altering genetic support predicts
approval only when the target is not broadly pleiotropic, and that conditional
pattern reproduces in an independently curated resource at P = 0.001._

Two limits to state honestly. The interaction is specific to therapeutic-area
breadth in ChEMBL — in gPS it is absent there at every cut point (OR 0.62–1.25,
p ≥ 0.44) while present in Pharmaprojects at cuts ≥ 8 (OR 4.1–7.1). And on the
subset of Pharmaprojects pairs absent from ChEMBL entirely, 236 supported pairs
with 13 launches give OR 1.64 [0.17, 15.85], p = 0.67 — same direction, no
power, so it cannot be quoted either way.

This also explains the non-linearity null in section 3: that test used
continuous log-TA terms pooled over all supported pairs, the stratum where the
ceiling is weakest, and was underpowered besides.

## Correction needed in the manuscript

The manuscript states that 52 of 242 approved GWAS-supported target–indication
pairs (21.5%) meet the strict definition. The correct count is **51 (21.1%)**.
The published enrichment table reports `yes_evid-high_clinphase = 51`, and 51 is
the value consistent with every other published number: 32777 × 51 / (4513 × 36)
= 10.2890 (52 gives 10.4907), and the 2×2 cells sum to the published 37,377
pairs. The odds ratio, relative success and P value are unaffected.

## Open editorial decision

Whether the abstract, Results and Discussion keep OR 10.3 with the held-out
evidence attached, or move to the bias-corrected 8.63. The frozen definition
shows no out-of-sample degradation, so 10.3 is defensible as reported; 8.63 is
the honest figure _if_ the definition is treated as having been chosen by a
threshold search. Both are supported by the analysis; the choice is editorial.

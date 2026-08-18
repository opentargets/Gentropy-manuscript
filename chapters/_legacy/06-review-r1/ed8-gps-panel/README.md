# Extended Data Fig. 8 — gPS panel (referee R1-mn-14)

Referee 1: \*"Extended Data Figure 8: Shouldn't you be consistent and categorize
high pleiotropy as

> = 10 diseases, as you have done previously (Fig. 5b) and as supported by the
> model (Fig. 5c)? You don't include TA pleiotropy in Fig. 5b."\*

The Fig. 5b half is done in [`../fig5b-ta-stratum/`](../fig5b-ta-stratum/). This
is the other half: Extended Data Fig. 8 becomes a two-panel figure — **(a)** the
published therapeutic-area grouping, **(b)** the same analysis grouped by gPS —
so it can be read on either scale.

## Headline: the pattern only partly holds on gPS

**The Phase II→III signal replicates on gPS and is the strongest effect there
too.** But two other transitions that were null on the therapeutic-area scale
become significant on gPS, and one comparison that was significant on TAs is not
on gPS:

| Comparison                        | TAs panel (published) | gPS panel                         |
| --------------------------------- | --------------------- | --------------------------------- |
| Phase I→II, Medium vs Low         | 0.058 — null          | **0.0027 — significant**          |
| Phase II→III, Medium vs Low       | 2.7 × 10⁻⁴            | 0.0047                            |
| Phase II→III, High vs Low         | 5.9 × 10⁻⁷            | 1.6 × 10⁻⁴                        |
| Phase II→III, High vs Medium      | 0.010                 | 0.073 — **no longer significant** |
| Phase III→approval, Medium vs Low | 0.136 — null          | **0.041 — significant**           |

BH-adjusted P, nine tests per panel.

**This contradicts the main text as written.** §06 currently says: _"Phase
transition probabilities differed across pleiotropy groups, defined here by the
number of TAs, primarily at the Phase II–III transition […] Other phase
transitions (Phase I–II: 82.6–84.2%; Phase III–approval: 29.4–31.2%) did not
differ significantly across pleiotropy groups after correction for multiple
testing."_ Once panel (b) is in the figure, a reader sees brackets on all three
transitions. The "defined here by the number of TAs" clause scopes the first
sentence, but the "Other phase transitions … did not differ significantly"
sentence reads as general and is false on the gPS scale. That sentence needs a
main-text change. Flagging only — no text was drafted or edited, per
instructions.

Two further points worth having in hand for that edit:

- Both newly significant gPS effects are **Medium vs Low only**, and both run
  _in favour of_ Medium: Phase I→II 84.4% (Medium) vs 82.3% (Low), RR 1.03;
  Phase III→approval 31.7% vs 29.0%, RR 1.09. So they do not weaken the paper's
  story — they are the same intermediate-pleiotropy optimum that Fig. 5c shows,
  now visible at two more transitions. Neither High-vs-Low comparison is
  significant at those transitions (0.194 and 0.784), so the _monotone_ "more
  pleiotropy is worse" reading remains confined to Phase II→III.
- The loss of Phase II→III High vs Medium on gPS is a resolution effect, not a
  reversal. The gPS Medium bin (2–9) is wider than the TA Medium bin and its
  High bin starts at gPS ≥ 10, so the Medium/High contrast spans a smaller gap
  on the gPS scale: rates 53.8% vs 51.6% (RR 0.96) against 53.5% vs 49.8% (RR
  0.93) on TAs. Direction is unchanged.

## Step 1 control — passed

Universe `uniqueTherapeuticAreas >= 1`: **18,480 pairs**, binned **6,578 / 9,705
/ 2,197** for Low (1 TA) / Medium (2–5) / High (≥ 6). Every printed value
reproduces:

| Check                             | Expected                           | Got                            |
| --------------------------------- | ---------------------------------- | ------------------------------ |
| Phase I→II, Low / Medium          | 82.9% / 84.2%                      | 82.9% / 84.2%                  |
| Phase II→III, Low / Medium / High | 57.0% / 53.5% / 49.8%              | 57.0% / 53.5% / 49.8%          |
| Phase III→approval, Low / Medium  | 29.4% / 31.2%                      | 29.4% / 31.2%                  |
| RR Medium vs Low, II→III          | 0.94 (0.91–0.97), P_adj = 3 × 10⁻⁴ | 0.94 (0.91–0.97), 2.706 × 10⁻⁴ |
| RR High vs Low, II→III            | 0.87 (0.83–0.92), P_adj < 0.001    | 0.87 (0.83–0.92), 5.900 × 10⁻⁷ |
| RR High vs Medium, II→III         | 0.93 (0.88–0.98), P_adj = 0.010    | 0.93 (0.88–0.98), 1.030 × 10⁻² |

These are hard `assert`s in the notebook, so the control cannot silently rot. As
an independent check, re-running the published plotting cell reproduces
`extended_figure_8.pdf` byte-for-byte apart from the embedded `/CreationDate`.

## BH convention

Fig. 5b's printed FDRs use plain BH (`p · m / rank`, no monotone enforcement) —
established in [`../fig5b-ta-stratum/`](../fig5b-ta-stratum/). **Extended Data
Fig. 8 is different: the published values come from
`statsmodels.multipletests(method="fdr_bh")`, which does enforce monotonicity.**

For this family the distinction does not matter. On the TA panel the two
conventions differ in exactly one of nine tests — Phase III→approval High vs
Medium, 0.759 (monotone) vs 0.788 (plain) — which is not printed anywhere. On
the gPS panel they agree on all nine. No significance call at 0.05 changes under
either convention, on either panel. The monotone form is used because it is what
the published notebook ran; both columns (`p_adj_bh`, `p_adj_bh_plain`) are
carried into the test CSV.

## Panel (a) — therapeutic areas (published, unchanged)

Group sizes 6,578 / 9,705 / 2,197.

| Transition         | 1 TA                   | 2–5 TAs | ≥ 6 TAs |
| ------------------ | ---------------------- | ------- | ------- |
| Phase I→II         | 82.9% (5,453 of 6,578) | 84.2%   | 82.6%   |
| Phase II→III       | 57.0%                  | 53.5%   | 49.8%   |
| Phase III→approval | 29.4%                  | 31.2%   | 30.6%   |

| Transition         | Comparison     | RR (95% CI)      | P raw       | P_adj           |
| ------------------ | -------------- | ---------------- | ----------- | --------------- |
| Phase I→II         | Medium vs Low  | 1.02 (1.00–1.03) | 2.57 × 10⁻² | 5.78 × 10⁻²     |
| Phase I→II         | High vs Low    | 1.00 (0.97–1.02) | 7.59 × 10⁻¹ | 7.59 × 10⁻¹     |
| Phase I→II         | High vs Medium | 0.98 (0.96–1.00) | 6.50 × 10⁻² | 1.17 × 10⁻¹     |
| Phase II→III       | Medium vs Low  | 0.94 (0.91–0.97) | 6.01 × 10⁻⁵ | **2.71 × 10⁻⁴** |
| Phase II→III       | High vs Low    | 0.87 (0.83–0.92) | 6.56 × 10⁻⁸ | **5.90 × 10⁻⁷** |
| Phase II→III       | High vs Medium | 0.93 (0.88–0.98) | 3.43 × 10⁻³ | **1.03 × 10⁻²** |
| Phase III→approval | Medium vs Low  | 1.06 (0.99–1.14) | 9.06 × 10⁻² | 1.36 × 10⁻¹     |
| Phase III→approval | High vs Low    | 1.04 (0.93–1.16) | 4.96 × 10⁻¹ | 6.38 × 10⁻¹     |
| Phase III→approval | High vs Medium | 0.98 (0.88–1.09) | 7.00 × 10⁻¹ | 7.59 × 10⁻¹     |

## Panel (b) — gPS

Same universe (18,480 pairs), regrouped on `uniqueDiseases`. Group sizes **5,724
/ 9,542 / 3,214**, confirmed.

| Transition         | gPS 1 | gPS 2–9 | gPS ≥ 10 |
| ------------------ | ----- | ------- | -------- |
| Phase I→II         | 82.3% | 84.4%   | 83.4%    |
| Phase II→III       | 56.7% | 53.8%   | 51.6%    |
| Phase III→approval | 29.0% | 31.7%   | 29.4%    |

Wilson 95% intervals and the n at start / n reaching for every cell are in
`ed8_phase_transition_rates-r1.csv`.

| Transition         | Comparison     | RR (95% CI)      | P raw       | P_adj           |
| ------------------ | -------------- | ---------------- | ----------- | --------------- |
| Phase I→II         | Medium vs Low  | 1.03 (1.01–1.04) | 6.05 × 10⁻⁴ | **2.72 × 10⁻³** |
| Phase I→II         | High vs Low    | 1.01 (0.99–1.03) | 1.51 × 10⁻¹ | 1.94 × 10⁻¹     |
| Phase I→II         | High vs Medium | 0.99 (0.97–1.01) | 2.13 × 10⁻¹ | 2.40 × 10⁻¹     |
| Phase II→III       | Medium vs Low  | 0.95 (0.92–0.98) | 1.55 × 10⁻³ | **4.66 × 10⁻³** |
| Phase II→III       | High vs Low    | 0.91 (0.87–0.95) | 1.77 × 10⁻⁵ | **1.59 × 10⁻⁴** |
| Phase II→III       | High vs Medium | 0.96 (0.92–1.00) | 4.06 × 10⁻² | 7.30 × 10⁻²     |
| Phase III→approval | Medium vs Low  | 1.09 (1.01–1.18) | 1.81 × 10⁻² | **4.08 × 10⁻²** |
| Phase III→approval | High vs Low    | 1.01 (0.92–1.12) | 7.84 × 10⁻¹ | 7.84 × 10⁻¹     |
| Phase III→approval | High vs Medium | 0.93 (0.85–1.02) | 1.13 × 10⁻¹ | 1.70 × 10⁻¹     |

## Sensitivity — gPS ≤ 5 / 6–9 / ≥ 10 (not in the figure)

Recorded so the bins can be switched if something looks wrong. Group sizes
12,544 / 2,722 / 3,214.

| Transition         | gPS ≤ 5 | gPS 6–9 | gPS ≥ 10 |
| ------------------ | ------- | ------- | -------- |
| Phase I→II         | 83.4%   | 84.5%   | 83.4%    |
| Phase II→III       | 54.8%   | 55.6%   | 51.6%    |
| Phase III→approval | 30.8%   | 30.2%   | 29.4%    |

Only Phase II→III survives correction, and only against the ≥ 10 group: High vs
Low RR 0.94 (0.90–0.98), P_adj = 0.019; High vs Medium RR 0.93 (0.88–0.98),
P_adj = 0.019. Medium vs Low is flat (RR 1.02, P_adj = 0.667), and nothing at
Phase I→II or Phase III→approval is significant (smallest P_adj = 0.413).

So these bins give the cleanest match to the published claim — II→III only,
driven by the ≥ 10 group. The trade-off is that they collapse the low end:
`gPS ≤ 5` is 12,544 of 18,480 pairs (68%), which merges the gPS-1 tier that
carries the intermediate-optimum signal into the Low bin and hides it. Under
this convention `p_adj_bh` and `p_adj_bh_plain` diverge in one printed cell
(II→III High vs Low: 0.0187 monotone vs 0.0282 plain), though the significance
call is unchanged.

Main bins were kept for the figure: ≥ 10 for High is what the referee asked for
and what Fig. 5b uses, and a Low bin of gPS 1 keeps the Low tier comparable to
the 1-TA Low tier in panel (a).

## Thin cells

Fewest pairs at the start phase, all at Phase III→approval:

| Grouping        | Group    | n at start | n reaching |
| --------------- | -------- | ---------- | ---------- |
| TAs             | ≥ 6 TAs  | 903        | 276        |
| gPS             | gPS ≥ 10 | 1,383      | 407        |
| gPS sensitivity | gPS 6–9  | 1,280      | 387        |

None is small enough to threaten the normal approximation in the z-test (all
expected counts > 250), but the Phase III→approval Wilson intervals are
correspondingly the widest in the figure — the ≥ 6 TAs interval spans
27.6–33.6%. The two newly significant gPS effects are not in these cells: Phase
III→approval Medium vs Low compares n = 4,335 against n = 2,671.

## Figure

`chapters/03-manuscript-figures/extended_figures/extended_figure_8-r1.pdf`

Two panels, labelled **a** (TAs) and **b** (gPS), sharing colours (`#c6dbef` /
`#6baed6` / `#08519c`), fonts, bar geometry and y-axis range (0.2–0.97,
`sharey`). Group n above each bar. Significance brackets only where P_adj <
0.05: adjacent pairs on the lower level, the spanning Low–High bracket above.

One shared figure-level legend. Because the bins differ between panels, each of
the three legend entries names the tier and both definitions —
`Low — 1 TA (a) / gPS 1 (b)`, and so on. This satisfies "one shared legend" and
"group labels must state the metric" together; the alternative, a per-panel
legend, would have been two legends, and a six-entry legend would repeat all
three colours.

The published `extended_figure_8.pdf` is untouched. The published plotting cell
now writes `extended_figure_8_control_ta_only-r1.pdf`, kept as the control
render.

## Run

```bash
# rebuild the appended notebook section (idempotent; appended cells are tagged r1_ed8_gps)
.venv/bin/python chapters/06-review-r1/ed8-gps-panel/build_notebook.py

# execute
cd chapters/03-manuscript-figures/extended_figures
../../../.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 ed8_translation_success_by_pleiotropy.ipynb
```

The notebook is the deliverable; `build_notebook.py` exists so the appended
section can be regenerated without hand-editing JSON.

## Outputs

| File                                                                                          | Contents                                                                             |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `chapters/03-manuscript-figures/extended_figures/ed8_translation_success_by_pleiotropy.ipynb` | extended notebook, control assertions plus both panels                               |
| `chapters/03-manuscript-figures/extended_figures/extended_figure_8-r1.pdf`                    | two-panel figure                                                                     |
| `chapters/03-manuscript-figures/extended_figures/extended_figure_8_control_ta_only-r1.pdf`    | control render of the published single panel                                         |
| `data/intermediate_files/ed8_phase_transition_rates-r1.csv`                                   | 27 rows — per group per transition: group n, n at start, n reaching, rate, Wilson CI |
| `data/intermediate_files/ed8_phase_transition_tests-r1.csv`                                   | 27 rows — per pairwise test: both rates, risk ratio, CI, raw P, both BH conventions  |

Both CSVs cover all three groupings (`TAs`, `gPS`, `gPS_sensitivity`) via the
`grouping` column.

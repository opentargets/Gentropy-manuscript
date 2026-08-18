# Fig. 5b — number-of-TAs pleiotropy group (referee R2-MJ-1)

Adds the second pleiotropy axis to Fig. 5b so the number of therapeutic areas
(TAs) is shown next to gPS, and states the corresponding contrast in the Results
text.

## Run

```bash
# 1. statistics and the augmented forest-plot input
.venv/bin/python chapters/06-review-r1/fig5b-ta-stratum/fig5b_ta_contrast.py

# 2. regenerate the figure (run from the figure directory; the root .Rprofile activates the root renv)
cd chapters/03-manuscript-figures/figure_5
R_LIBS_SITE="$(git rev-parse --show-toplevel)/chapters/03-manuscript-figures/renv/library/macos/R-4.5/aarch64-apple-darwin25.0.0" \
  Rscript figure_5.R
# -> chapters/03-manuscript-figures/figure_5/figure_5_final-r1.pdf
```

## New forest rows

`any` genetic support, PAV not required, phase 4+, from
`ti_pairs_chembl_master-r1.parquet` (37,377 pairs, 4,564 approved). Both rows
reproduce `or10_phase0_grid_full-r1.csv` exactly, and the widest window (`any` +
TA >= 1) returns 742 supported pairs with 242 approved, matching the main text.

| Definition    | OR (95% CI)      | RS   | not approved / approved |
| ------------- | ---------------- | ---- | ----------------------- |
| any + TA = 1  | 4.29 (2.53–7.28) | 3.06 | 37 / 22                 |
| any + TA >= 6 | 2.89 (2.23–3.74) | 2.35 | 204 / 81                |

## Contrast

The published panel fits one logistic model over all pairs with "no genetic
support" as the common reference and Wald-tests the two supported strata against
each other. The reference coefficient enters both stratum log-odds identically
and cancels, so the fitted contrast equals the closed-form log odds ratio of the
two supported 2×2 rows. Both are computed and agree to machine precision.

| Contrast          | log OR | SE     | z     | P          |
| ----------------- | ------ | ------ | ----- | ---------- |
| TA 1 vs >= 6      | 0.4038 | 0.2995 | 1.348 | **0.1777** |
| gPS <= 5 vs >= 10 | 0.4805 | 0.1803 | 2.664 | 0.00772    |

The gPS row reproduces the published $P = 0.008$, which confirms the method. The
TA contrast does not reach significance. This is not in tension with panel c,
where the continuous non-linear TA model is _stronger_ than the gPS one: a
two-bin split discards most of the information the quadratic fit uses. The
alternative low bin (TA 2–5 vs >= 6) is also not significant, so no choice of
cut point rescues it — hence the number is stated in the text rather than left
as a missing asterisk.

Note the published bracket P values in `drug_enrichment_subsets_vs_full_l2g.csv`
(`diffence_pval`, gPS 0.0163 / 0.0149) come from each stratum against its
_complement_, not from this low-versus-high contrast; the manuscript reports the
contrast, so this script recomputes it.

## FDR family

The published FDRs (PAV 0.001, rare 0.015, gPS 0.01, effect size 0.19) are plain
step-up BH, `p * m / rank`, _without_ the usual monotone enforcement — which is
why the published rare FDR (0.015) exceeds the published gPS FDR (0.01) despite
near-identical raw P values. The script keeps that convention so the printed
values stay reproducible, and also reports the textbook monotone version.

| Comparison        | raw P    | FDR now (m = 4) | FDR after (m = 5) |
| ----------------- | -------- | --------------- | ----------------- |
| PAV vs non-PAV    | 0.000244 | 0.001           | 0.001             |
| rare vs common    | 0.007717 | 0.015           | **0.019**         |
| gPS <= 5 vs >= 10 | 0.007718 | 0.010           | **0.013**         |
| effect size       | 0.193    | 0.193           | 0.193             |
| TA 1 vs >= 6      | 0.178    | —               | 0.222             |

Nothing crosses 0.05. Two printed values change: the rare FDR and the gPS FDR.
The effect-size comparison prints only its raw P in the text, and that raw P is
unchanged, so the effect-size sentence is untouched — its own FDR is
`0.193313 * 4/4 = 0.193313 * 5/5`, identical before and after.

The rare and gPS raw P values differ only at the sixth significant digit
(0.007717358 vs 0.007717500), so rare keeps rank 2 and gPS rank 3, as in the
published family. Any future change that flips that order would swap the two
printed FDRs.

## Figure changes

`chapters/03-manuscript-figures/figure_5/figure_5.R`:

- reads `drug_enrichment_subsets_vs_full_l2g-r1.csv` when present, else the
  published table
- new `TAs` facet with `TAs>=6` / `TAs=1`, placed directly below the gPS facet
- `category_order` now spells out the published facet order explicitly. The
  published panel supplied the facet variable as a factor in the point layer and
  as a character in the significance-label layer, so ggplot silently fell back
  to collation order; `panel_sig_df$category` is now a factor on the same
  levels, which makes the order intentional. Every other facet keeps its
  published position.
- no asterisk on the `TAs` facet, matching the effect-size facet (P = 0.19),
  which is also unmarked
- output is `figure_5_final-r1.pdf`; the published `figure_5_final.pdf` is not
  overwritten

## Manuscript text

`06_therapeutic_success.diff` holds the three edits to
`~/Projects/manuscript_gentropy/sections/results/06_therapeutic_success.tex`
(rare FDR, gPS FDR plus the new TA sentence, caption). That repo is read-only
from here, so apply it there:

```bash
cd ~/Projects/manuscript_gentropy
git apply ~/Projects/Gentropy-manuscript/chapters/06-review-r1/fig5b-ta-stratum/06_therapeutic_success.diff
```

Verified with `git apply --check`.

## Outputs

| File                                                                 | Contents                                            |
| -------------------------------------------------------------------- | --------------------------------------------------- |
| `data/intermediate_files/fig5b_ta_rows-r1.csv`                       | the two new forest rows                             |
| `data/intermediate_files/fig5b_ta_contrast-r1.csv`                   | both within-group contrasts, closed form and fitted |
| `data/intermediate_files/fig5b_fdr_family-r1.csv`                    | BH FDR before and after, plain and monotone         |
| `data/intermediate_files/drug_enrichment_subsets_vs_full_l2g-r1.csv` | published table plus the two TA rows                |
| `chapters/03-manuscript-figures/figure_5/figure_5_final-r1.pdf`      | regenerated figure                                  |

# Figure 4 — panel c on an odds-ratio axis, panel a on the resolved trait column

Reviewer 2, minor comment 3: _"I suggest to report OR's rather than log(OR)'s
for the gPS. They're much more intuitive to most readers."_ The in-text values
and the Figure 4 caption were already converted to odds ratios per doubling of
gPS; only the plotted axis still showed log-odds, so the caption and the panel
described different scales.

| File                             | What                                                                                                             |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `figure_4-r1.R`                  | The figure script. Differs from `figure_4.R` in the panel a input, the panel c axis, and the output path.        |
| `figure_4_final-r1.pdf`          | Output.                                                                                                          |
| `figure_4_control-r1.R` / `.pdf` | `figure_4.R` with only the output path changed, used for step 1.                                                 |
| `../_run_r1.R`                   | Two-line wrapper: `renv` activates in `chapters/03-manuscript-figures`, `figure_4.R` needs the repo root as cwd. |

Run from `chapters/03-manuscript-figures`:
`Rscript _run_r1.R chapters/03-manuscript-figures/figure_4/figure_4-r1.R`

## Step 0 — staged input

`data/intermediate_files/gene_pleiotropy_by_category-r1.csv` was copied to
`data/figure_4/gene_pleiotropy_by_category.csv`, which `figure_4.R:213`
hard-reads. The `-r1` original is untouched. That table was recovered by
`chapters/06-review-r1/cluster-disease-vs-ta/06_panel_c_recovery-r1.py`; it is
not in the repo because `04_gene_pleiotropy_by categories.ipynb` computes it and
never writes it.

## Step 1 — control, all three panels: PASS

`figure_4_control-r1.R` is `figure_4.R` with a single line changed — the output
filename — so the published `figure_4_final.pdf` was never at risk of being
overwritten by the control run. Its md5 (`e858b1cc61a26ce3fed8bcd323182823`) was
recorded before and confirmed after.

- **Text layer byte-identical** to the published PDF under `pdftotext -layout`.
  That covers all 21 category labels, all 21 gene counts, all 21 overlap
  percentages, the row ordering — including the 0.0006-wide Cell non-essential /
  Cell essential inversion — and every axis label in all three panels. The 42
  assert targets are therefore met exactly.
- **Zero differing pixels** across the whole figure at 3600 px wide: panel a 0,
  panel b 0, panel c 0. No R-version glyph drift of the kind seen on Figure 2.

The recovered panel c input rebuilds the published figure exactly, so it is
trustworthy.

## Step 2 — panel a on the resolved vPS series

Input swapped to `Fig4A_stats_variant_pleiotropy-r1.csv` (2006–2024, the same 19
years as the committed file). gPS and variants-per-gene inputs are unchanged —
both already read `diseaseIds`.

| Measure                                       | Value                                       |
| --------------------------------------------- | ------------------------------------------- |
| Differing pixels, panel a                     | 2,145 of 1,349,095 (**0.159%**)             |
| Bounding box                                  | y 442–535, x 488–713 of a 1045 × 1291 panel |
| vPS line shift                                | mean **+0.230 px**, max +4 px in one column |
| … right quarter of the x-range (~2019 onward) | mean **+0.790 px**                          |
| vPS stroke width                              | 3 px                                        |
| gPS line                                      | **pixel-identical** (2,014 px both)         |
| Variants-per-gene line                        | **pixel-identical** (1,252 px both)         |

Within the expected envelope: the line moves about a quarter of its own stroke
width, only from ~2019 rightward, and nothing else in the panel moves.

## Step 3 — panel c transformation

A column swap, not a recomputation. `odds_ratio`, `ci_lower` and `ci_upper` are
already in the recovered table — they are `exp()` of the log columns, written by
the source notebook. `fdr` still comes from `p.adjust` at line 225.

- `aes(x = odds_ratio)`, `geom_errorbar(xmin = ci_lower, xmax = ci_upper)`
- `geom_vline(xintercept = 1)`, moved from 0
- `scale_x_log10()`, so geometry is preserved: position on a log OR axis is
  proportional to log(OR) = `log_odds_ratio`, the previously plotted value
- Axis label `Odds ratio per doubling of gPS`
- Ticks **0.8, 0.9, 1, 1.1, 1.25, 1.5** — the user-suggested set, symmetric
  about 1 on a log scale (0.8 and 1.25 are exact reciprocals; 0.9 and 1.1 are
  within 1% of reciprocal). Labels are written literally so the reference tick
  reads `1`, not `1.00`. The data span OR 0.733–1.627, so every tick falls
  inside the plotted range.
- Row order, Genes and In set annotations, and the blue/grey FDR colouring are
  untouched. 15 of 21 categories remain significant at FDR 5%.

### Layout constants

All panel c layout constants are still derived in log space and then
exponentiated, because that is what keeps the geometry identical:

```r
x_col1 <- exp(x_hi + x_span * 0.22)
x_col2 <- exp(x_hi + x_span * 0.42)
x_axis_lo <- exp(x_lo - 0.05 * x_span)
x_axis_hi <- exp(x_hi + 0.05 * x_span)
```

One fix was needed beyond the swap. The `Gene/Target Set` header was annotated
at `x = -Inf`; `log10(-Inf)` is `NaN`, so ggplot silently dropped the layer and
the header vanished. `-Inf` resolves to the coord range _after_ the default 5%
continuous expansion, not to `x_axis_lo`, so that bound is now reproduced
explicitly in log space:

```r
x_hdr_anchor <- 10^(log10(x_axis_lo) - 0.05 * (log10(x_axis_hi) - log10(x_axis_lo)))
```

With that, the header lands on exactly its published pixel.

## Verification — panel c points are unmoved

Detecting the coloured marks in panel c and grouping them into bands gives 25
wide bands: the 21 category rows plus the header row, the axis line, the tick
labels and the axis title.

**22 of 25 bands have byte-identical x extent, including all 21 data rows and
the header.** The only three that differ are the axis furniture:

| Band | y range   | What                             | Δ left / right |
| ---- | --------- | -------------------------------- | -------------- |
| 22   | 1117–1128 | axis line and ticks              | +0 / +4 px     |
| 23   | 1150–1174 | tick labels (−0.2…0.4 → 0.8…1.5) | −16 / +5 px    |
| 24   | 1216–1245 | axis title                       | −4 / +4 px     |

Every point and every confidence interval sits on exactly the pixel it occupied
in the published panel. Panel c's 21,876 differing pixels are entirely the
relabelled axis.

Panel b is byte-identical: **0 differing pixels**.

The 42 assert targets were re-checked after the change — all 21 gene counts, all
21 percentages and the row order are unchanged in the `-r1` text layer.

# Rare-variant share of discovery over time — Extended Data Fig. 10

Reviewer 1, minor comment 3.

## The comment (verbatim)

> Lines 99-101: It would be helpful to call out the increase in the contribution
> of rare variants to gene discovery. At the scale of the y axis, i.e. relative
> to common variants, it is difficult to see the increasing role of rare
> variants in discovery (assuming there is such). A supplementary figure, or
> addition to Fig. 1, that specifically shows that increase over time, would be
> a helpful contribution. As well as an addition to the discussion around what
> is leading to that increase (i.e. sample size, imputation, WGS, etc.).

Rare credible sets are 15,311 of 520,975 qualifying credible sets (2.94%), so on
Figure 1c's shared y-axis the rare band is a sliver and the trend is invisible.
One figure showing the rare share on its own percent scale is the whole answer.
The drivers half of the comment (sample size, imputation, WGS) is handled by
Discussion wording, not by analysis.

**Scope is one figure.** No analyses are added.

## Definition — state it next to any number, do not paraphrase it

The rare layer is entities **not reachable from any common-variant study** —
those that would not have been found without rare variants. It is a reachability
difference between the nested tiers of Figure 1c, **not** "genes first
identified through rare variants". The published ancestry numbers (2,462 /
16,384) are the same construction, and an earlier draft misdescribed them as
first-discovery counts and had to be corrected.

## What is plotted

Two panels, both cumulative by year, 2006–2024, bars in the Figure 1c rare
colour, one shared percent y-axis range:

- **a** — rare-variant share of cumulative disease-associated genes (%)
- **b** — rare-variant share of cumulative gene–disease associations (%)

For each metric and year, from `fig1c_cumulative_discovery_nested-r1.csv` alone:

```
share = layer[layer_label == "rare"] / cumulative[tier_index == 4] * 100
```

Both quantities are already cumulative in the input, so there is no differencing
and no other file is involved.

The two panels are given the **same** y range (0–2.0%) so the genes and
associations series are directly comparable by eye, which is the point of
putting them side by side.

## Results

|                                       | disease genes                    | gene–disease associations              |
| ------------------------------------- | -------------------------------- | -------------------------------------- |
| Rare-only, 2024                       | 115                              | 630                                    |
| All variants, cumulative 2024         | 8,129                            | 35,535                                 |
| **Rare share, 2024**                  | **1.41%**                        | **1.77%**                              |
| First year with a non-zero rare layer | 2011                             | 2011                                   |
| Share in that first year              | 0.18%                            | 0.14%                                  |
| Change to 2024                        | +1.23 pp (7.8×)                  | +1.63 pp (12.8×)                       |
| Monotone increasing?                  | **no** — decreases in 2019, 2023 | **no** — decreases in 2016, 2018, 2019 |

So the increase the referee suspected is real: the rare-variant share of
cumulative discovery rises roughly eight-fold (genes) and thirteen-fold
(associations) between 2011 and 2024. It is not monotone — two and three
single-year dips respectively, each because a large common-variant year grows
the denominator faster than the rare layer grows. The safe wording is "increases
steadily, though not in every year", not "increases every year".

## Early years, where the denominator is small

The denominator is a cumulative total, so it is never zero: the share is
**defined in every year of the axis**, and no year is dropped, imputed, smoothed
or clipped. It is nonetheless tiny at the start — 1 entity in 2006 — and the
rare layer is exactly 0 until 2011, so 2006–2010 are true zeros rather than
missing data and are plotted as zero-height bars. Because the quantity is
cumulative rather than annual, the small-denominator years are only unstable at
the very left edge; by 2011, where the first non-zero bar appears, the
denominator is already 552 genes / 723 associations. The
`all_variants_cumulative` column is exported so a reader can see the denominator
behind every point.

## Conventions kept unchanged

- **FinnGen R12 carries no publication date in the 25.06 release; this project
  pins it to 2024-11-04**, following
  `ancestry-mixed-split/01_ancestry_reclassification.ipynb`. That places FinnGen
  — the largest single contributor of rare-variant credible sets — in the
  **final year of the axis**, which is why 2024 is the largest single step in
  both panels. The convention is not changed here.
- **Year axis 2006–2024.** `MAX_YEAR = 2024`; 2025 is a partial year and is
  excluded project-wide.
- **Rare credible sets pass extra QC by design** — a rare credible set is
  retained only if it replicates, or colocalises with a molQTL, or carries a
  protein-altering variant. Both shares are therefore conservative. Intended,
  already disclosed in the manuscript, and neither relaxed nor quantified here.

## Input

`data/intermediate_files/fig1c_cumulative_discovery_nested-r1.csv`, produced by
`chapters/06-review-r1/ancestry-mixed-split/01_ancestry_reclassification.ipynb`.
Nesting order EUR → non-EUR → mixed (all common) → rare (any ancestry),
`tier_index` 1–4. Nothing is recomputed.

## Outputs

| File                                                              | Contents                                                                                                                                                                                            |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `extended_figure_10.pdf` (this directory)                         | The figure, vector, 7.2 × 3.2 in                                                                                                                                                                    |
| `~/Projects/manuscript_gentropy/figures/extended_figure_10.pdf`   | Same file, copied into the paper source. New file; nothing there was overwritten                                                                                                                    |
| `data/intermediate_files/rare_discovery_over_time-r1.csv`         | The plotted values: one row per metric × year with `rare_cumulative`, `all_variants_cumulative`, `rare_share_pct`, and a `definition` column carrying the wording the numbers must be reported with |
| `data/intermediate_files/rare_discovery_over_time_summary-r1.csv` | One row per metric: first non-zero year, endpoint shares, change and ratio, monotonicity and which years decreased, smallest denominator                                                            |

## How to run

Notebook (pandas only; no Spark, seconds):

```bash
cd chapters/06-review-r1/rare-variant-discovery-over-time
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 01_rare_discovery_over_time.ipynb
```

Figure (R + ggplot2 + patchwork). This directory has no `renv` project of its
own, so borrow the figures library rather than adding one:

```bash
cd chapters/06-review-r1/rare-variant-discovery-over-time
R_LIBS_SITE="$(git rev-parse --show-toplevel)/chapters/03-manuscript-figures/renv/library/macos/R-4.5/aarch64-apple-darwin25.0.0" \
  Rscript ed10_rare_variant_discovery.R
cp extended_figure_10.pdf ~/Projects/manuscript_gentropy/figures/extended_figure_10.pdf
```

Environment gotchas:

- The library path is platform-specific
  (`macos/R-4.5/aarch64-apple-darwin25.0.0`). On another machine, run from
  `chapters/03-manuscript-figures/` so its `.Rprofile` activates `renv`
  normally, and pass the script by absolute path. Do not add an `.Rprofile` here
  — nothing under `chapters/03-manuscript-figures/` may be modified, and a
  second `renv` project would duplicate it.
- Running the script from the **repo root** would source the root `.Rprofile`,
  which activates the root `renv` project (R 4.3.1 lockfile, packages not
  installed). Run it from this directory, or pass `--no-init-file`.
- The PDF is written with `cairo_pdf`, so the figure ships as vector, not
  raster.

## Figure sizing and style

7.2 × 3.2 in, sized for
`\includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]`.

Matches Figure 1c / ED Fig 3: `theme_minimal()` with `#434343` text, `#8a8a8a`
axis lines, no grid, x ticks every 2 years with minor ticks between, bars rather
than lines. Rare colour `#FFC000`, the `rare` entry of `ANCESTRY_COLORS` in the
reclassification notebook — the same colour the rare band carries in Figure 1c.

## Known issues left alone

- 2024 is the largest step in both panels, partly because FinnGen R12 is pinned
  to it. Noted above rather than adjusted.
- The share dips in a few single years (listed above). Left visible; smoothing
  would hide exactly the kind of year-to-year variation a referee is entitled to
  see.
- No trend line or fitted model is drawn. The figure shows the series; it does
  not claim a functional form for the increase.

# Figure 2 (revision 1) and Extended Data Fig. 9

Reviewer 2, minor comment 2 asked for the MAF-versus-effect-size material to be
cut. The decision was to shorten the text and **demote** the panel rather than
delete it, so published panel a becomes a standalone Extended Data Fig. 9 and
Figure 2 keeps three panels.

Script: `figure_2-r1.R` (copy of `figure_2.R`; everything above the
`# ---- Combine plots horizontally using cowplot ----` section is unchanged).
`figure_2.R` and `figure_2_final.pdf` are untouched.

Run from this directory:

```
R_LIBS_SITE="$(git rev-parse --show-toplevel)/chapters/03-manuscript-figures/renv/library/macos/R-4.5/aarch64-apple-darwin25.0.0" Rscript figure_2-r1.R
```

## Outputs

| file                                           | page size                                   |
| ---------------------------------------------- | ------------------------------------------- |
| `figure_2_final-r1.pdf`                        | 8.27 x 2.5 in, same as `figure_2_final.pdf` |
| `../extended_figures/extended_figure_9-r1.pdf` | 7.19 x 3.19 in (517 x 229 pt)               |

## Control: does `figure_2.R` still rebuild the published figure?

Yes. `figure_2.R` was run unmodified before anything was changed and the result
was compared byte-for-byte with `figure_2_final.pdf` (which was restored
afterwards, md5 unchanged `ca9464baa59aa72de8f120bcf0204d9b`).

The PDF is not bit-reproducible, but the only differences are:

- `/CreationDate`, `/ModDate`;
- `/Producer (R 4.5.3)` published vs `(R 4.5.2)` locally;
- two `plotmath` text placements — the superscript `10` of the two
  `log10(mean MAF)` axis titles — shifted by 0.24 pt (`Tm ... 43.77` ->
  `44.01`), an R-version metric difference, and the xref offsets that shift with
  it.

Everything else in the decompressed content stream is identical. Rasterised at
300 dpi the two files differ in 326 of 1,860,000 pixels (0.018%), all inside
those two `10` glyphs. Control passed.

## What changed in the script

1. `plot_a` is dropped from the panel grid (the plot object is still built, it
   now feeds Extended Data Fig. 9).
2. `plot_grid` goes from four to three panels; `rel_widths`
   `c(1.2, 1.2, 1.2, 2)` -> `c(1.2, 1.2, 2)`, i.e. plot_a's 1.2 is removed and
   the remaining three keep their published proportions, so panel c keeps the
   wider slot.
3. Labels `c("a", "b", "c", "d")` -> `c("a", "b", "c")`; `label_x`
   `c(0, -0.02, -0.08, -0.04)` -> `c(0, -0.08, 0.09)`. plot*b takes slot 1
   (offset 0, as plot_a had) and plot_d keeps `-0.08`; plot_c's slot is wider
   than in the published grid, so its label needs a \_positive* offset to stay
   ~0.15 in from the panel (published gap 0.19 in) instead of drifting into the
   inter-panel gap.
4. `axis_title_a` and its cell in `plots_abcd_x_axes` removed; the x-axis-title
   row is now `axis_title_b, axis_title_d, axis_title_c` over the same three
   `rel_widths`.
5. Legend row: three cells, one per panel, and the study-type legend is now
   vertical. See below.
6. The four intermediate `figure_2_final_*.png` debug renders are not written by
   the r1 script; only the two deliverables are.
7. Extended Data Fig. 9 block added at the end.

No input data, colours, theme, fonts, `panel_aspect_ratio` (0.6), output width
or font sizes were changed.

## Panel mapping as built

| r1 panel             | plot object | content                                                                    |
| -------------------- | ----------- | -------------------------------------------------------------------------- |
| a                    | `plot_b`    | Proportion of PAV across MAF bins, 4 study types                           |
| b                    | `plot_d`    | mean \|beta\| across the five consequence groups, diseases vs measurements |
| c                    | `plot_c`    | consequence/localisation distribution across 4 study types                 |
| Extended Data Fig. 9 | `plot_a`    | mean \|beta\| vs MAF, 4 study types, 95% CI bands                          |

Verified against the render: panel a's plot area is 1.18 in wide with its x axis
at the same height as in the published figure, i.e. **identical in size to
published panel b** — `plot_b` and `plot_c` are pinned by `aspect.ratio = 0.6`,
so a wider slot only adds padding, it does not stretch them.

`plot_d` is the one panel with no `aspect.ratio` (by design: "width controlled
by `rel_widths` only"), so it absorbs the freed width: its plotted x axis goes
from 0.68 in to 1.16 in. The dots and intervals are unchanged, they are simply
spread over more width. If that is not wanted, `rel_widths <- c(1.2, 0.873, 2)`
reproduces plot_d's published 1.772 in slot exactly and moves the slack into the
gaps either side instead.

## The legend finding

Before: the study-type legend (`legend_ab`) was extracted from **`plot_a`** with
`get_legend()` and shared by panels a and b, because `plot_a` and `plot_b` carry
identical `scale_color_manual`/`scale_fill_manual` breaks and labels. Its cell
in the legend row was `rel_widths[1] + rel_widths[2]` wide, i.e. the two panels
merged (`rel_widths_ab_merged`). The x-axis titles were likewise stripped from
every panel (`axis.title.x = element_blank()`) and re-supplied as an inset row,
one cell of which came from `plot_a`.

So removing `plot_a` would have silently taken the study-type legend with it —
the legend does not belong to the panel it was drawn from.

After: `legend_ab` is taken from **`plot_b`**, which produces a byte-identical
legend (same scales, same `breaks`, same `labels`, same key sizes). Nothing was
restyled.

It is also **laid out vertically** (four rows, one entry each) instead of the
published single horizontal row. Serving one panel instead of two, it no longer
needs ~3.6 in of width; as a ~1.36 in wide block it sits under panel a and
leaves the space under panel b free for `legend_d`, which in the first
horizontal attempt was pushed out under panel c. Only the layout changed — same
entries, colours, key widths and 8 pt text. `legend.key.height` is 0.22 cm with
zero `legend.spacing.y` and zero legend margin, which is what makes four rows
fit in the height available.

Two knock-on positional fixes:

- The legend row cells are given in **inches of the 8.27 in canvas**,
  `c(0.604, 1.500, 1.945, 4.221)` for spacer / study-type / legend*d / empty.
  Cells simply equal to the panel slots do \_not* centre the legends on their
  panels — `legend_ab` lands left of panel a's axis and `legend_d` is thrown
  right of its cell by its own `legend.justification = c(1.16, 0.5)`, which put
  it under panel c. These widths are solved so each legend is centred on its
  panel's plotted axis. (In the first horizontal version the ~3.6 in grob was
  instead centred off the left edge of the canvas and clipped "cis-pQTL".)
- The legend inset drops from `0.15` to `0.11` of canvas height, because the
  vertical block is ~0.42 in tall rather than ~0.15 in and would otherwise sit
  on top of the `log10(mean MAF)` axis title.

Measured on the render: axis title a ends at y = 1.890 in; legend a occupies y
1.963-2.380 in, x 0.730-2.090 in, centre 1.410 in against panel a's axis centre
of 1.413 in; legend_d occupies y 2.027-2.300 in, x 3.207-4.187 in, centre 3.697
in against panel b's axis centre of 3.713 in. Nothing is clipped: all ink lies
within y 0.067-2.380 in of the 2.5 in canvas, and the two legends are 1.1 in
apart.

`legend_c` remains an empty `theme_void()` spacer: plot_c draws its own
consequence-category legend inside its panel (`legend.position = "right"`),
exactly as published. That is unaffected by the change.

## Extended Data Fig. 9

`plot_a` already carried its own x-axis title (`labs(x = log10(mean MAF))`) and
its own colour/fill scales; the published Figure 2 suppressed both
(`axis.title.x = element_blank()`, `legend.position = "none"`) and re-supplied
them from the grid assembly. Standalone it needs only:

- `legend.position = "right"` (vertical, 8 pt, same key sizes as the published
  legend row);
- the x-axis title centred (`hjust = 0.5` instead of the `0.8` that suited the
  packed grid);
- a plain plot margin.

Page size follows `extended_figure_10.pdf` (7.19 x 3.19 in), the closest sibling
Extended Data figure; `extended_figure_8-r1.pdf` is much larger (10.9 x 5.2 in)
so the narrower of the two references was used. Colours, theme, 8 pt fonts and
`aspect.ratio = 0.6` are as published.

### Caption check

The caption's ordering — measurements < diseases < cis-pQTLs < eQTLs — **holds
in all seven MAF bins** and is visible in the render:

| MAF bin   | measurement | disease | cis-pQTL | eQTL  |
| --------- | ----------- | ------- | -------- | ----- |
| 0.0-0.01  | 0.397       | 0.783   | 1.016    | 2.439 |
| 0.01-0.05 | 0.135       | 0.385   | 0.670    | 1.724 |
| 0.05-0.1  | 0.096       | 0.237   | 0.435    | 1.259 |
| 0.1-0.2   | 0.076       | 0.175   | 0.407    | 1.028 |
| 0.2-0.3   | 0.061       | 0.145   | 0.332    | 0.898 |
| 0.3-0.4   | 0.062       | 0.120   | 0.283    | 0.830 |
| 0.4-0.5   | 0.048       | 0.111   | 0.269    | 0.807 |

One caveat if the caption is worded strongly: in the lowest bin (MAF 0.0-0.01)
the cis-pQTL 95% CI (0.758-1.274) overlaps the disease CI (0.734-0.831), so
disease-versus-cis-pQTL is not separated there — the wide pink band crossing the
blue line at the left edge of the panel. All other adjacent pairs have
non-overlapping CIs in every bin. The four shaded 95% CI bands are all present,
though only the cis-pQTL one is wide enough to be obvious.

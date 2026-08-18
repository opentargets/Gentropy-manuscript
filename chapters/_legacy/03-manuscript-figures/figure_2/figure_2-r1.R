## Script to generate the revision-1 Figure 2 and Extended Data Fig. 9
# Derived from figure_2.R. Reviewer 2 minor comment 2 asked for the MAF-vs-effect-size
# material to be cut back, so the published panel a is demoted to a standalone
# Extended Data Fig. 9 and Figure 2 keeps three panels:
#   a = plot_b (proportion of PAV across MAF bins)
#   b = plot_d (mean |beta| across five consequence groups)
#   c = plot_c (consequence/localisation distribution across four study types)
# Outputs: figure_2_final-r1.pdf, ../extended_figures/extended_figure_9-r1.pdf
# Nothing above the "Combine plots" section differs from figure_2.R, and no input
# data are changed.
#
# The script requires the intermediate data files generated in
# chapters/02-analysis/02-variant-effects

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(cowplot)
  library(patchwork)
})

tryCatch(
  {
    setwd("chapters/03-manuscript-figures/figure_2")
  },
  error = function(e) {
    message(
      "Could not set working directory"
    )
  }
)


data_2a_path <- "data/figure_2_a.csv"
data_2b_path <- "data/figure_2_b.csv"
stopifnot(file.exists(data_2a_path))
stopifnot(file.exists(data_2b_path))

data_2a <- readr::read_csv(data_2a_path)
data_2b <- readr::read_csv(data_2b_path)


data_2c_path <- "data/figure_2_c.csv"
data_2d_path <- "data/figure_2_d.csv"
stopifnot(file.exists(data_2c_path))
stopifnot(file.exists(data_2d_path))

data_2c <- readr::read_csv(data_2c_path)
data_2d <- readr::read_csv(data_2d_path)


# Theme to mimic matplotlib styling (Helvetica, light grid, no spines)
base_theme <- theme_minimal() +
  theme(
    text = element_text(face = "plain", color = "#434343"),
    plot.title = element_text(
      face = "plain",
      size = 10,
      hjust = 0.5,
      color = "#434343"
    ),
    axis.title = element_text(size = 8, face = "plain", color = "#434343"),
    axis.text = element_text(size = 8, face = "plain", color = "#434343"),
    axis.text.x = element_text(
      size = 8,
      margin = margin(t = 0, b = 0),
      color = "#434343",
      angle = 45,
      hjust = 0.9
    ),
    axis.title.x = element_text(
      size = 8,
      face = "plain",
      color = "#434343",
      margin = margin(t = 10)
    ),
    axis.ticks = element_line(color = "#8a8a8a", linewidth = 0.2),
    axis.ticks.length = unit(0.08, "cm"),
    panel.background = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    axis.line = element_line(color = "#8a8a8a", linewidth = 0.3),
    legend.position = "right",
    legend.title = element_blank(),
    legend.text = element_text(face = "plain", color = "#434343", size = 8)
  )

# Define colors (shared by plot a and plot b)
colors <- c(
  "cis-pqtl" = "#A01813",
  "eqtl" = "#E08145",
  "gwas-disease" = "#245780",
  "gwas-measurement" = "#2F735F"
)

# Ensure studyType has a consistent order across both plots
studytype_levels <- c("cis-pqtl", "eqtl", "gwas-disease", "gwas-measurement")

# X-axis title position for plot a and b (hjust: 0 = left, 0.5 = center, 1 = right; vjust: vertical)
x_axis_title_hjust_a <- 0.8
x_axis_title_vjust_a <- 1
x_axis_title_hjust_b <- 0.8
x_axis_title_vjust_b <- 1

# Aspect ratio for all four panels (height/width; same value keeps panels uniform)
panel_aspect_ratio <- 0.6

# ---- Plot A ----

data_2a$studyType <- factor(
  data_2a$studyType,
  levels = studytype_levels
)

# X-axis breaks for plot a) (show all labels except 0.3-0.4)
x_breaks_a <- sort(unique(data_2a$mafBinMidpoint))
x_labels_a <- sort(unique(data_2a$mafBinRange))
x_labels_a[x_labels_a == "0.3-0.4"] <- ""

# Calculate y-axis limits for plot a) to ensure all data is visible
max_y_a <- max(data_2a$avgAbsEstimatedBetaInBucketCIUpper)
y_upper_a <- ceiling(max_y_a * 10) / 10 # Round up to nearest 0.1


# Create plot a)
plot_a <- ggplot(
  data_2a,
  aes(
    x = mafBinMidpoint,
    y = avgAbsEstimatedBetaInBucket,
    color = studyType,
    fill = studyType,
    group = studyType
  )
) +
  geom_ribbon(
    aes(
      ymin = avgAbsEstimatedBetaInBucketCILower,
      ymax = avgAbsEstimatedBetaInBucketCIUpper
    ),
    alpha = 0.12,
    colour = NA,
    na.rm = TRUE
  ) +
  geom_line(linewidth = 0.3, na.rm = TRUE) +
  scale_color_manual(
    values = colors,
    breaks = names(colors),
    labels = c("cis-pQTL", "eQTL", "GWAS (disease)", "GWAS (measurement)"),
    name = "studyType"
  ) +
  scale_fill_manual(
    values = colors,
    breaks = names(colors),
    labels = c("cis-pQTL", "eQTL", "GWAS (disease)", "GWAS (measurement)"),
    name = "studyType"
  ) +
  scale_x_log10(
    breaks = x_breaks_a,
    labels = x_labels_a,
    expand = c(0, 0)
  ) +
  labs(
    x = expression(log[10](mean~MAF)),
    y = expression(mean("|" * beta * "|"))
  ) +
  base_theme +
  coord_cartesian(ylim = c(0, y_upper_a)) +
  theme(
    plot.margin = margin(t = 5, r = 0, b = 15, l = 10),
    axis.title.x = element_text(
      size = 8,
      face = "plain",
      color = "#434343",
      margin = margin(t = 4),
      hjust = x_axis_title_hjust_a,
      vjust = x_axis_title_vjust_a
    ),
    axis.title.y = element_text(size = 8, face = "plain", color = "#434343"),
    legend.position = "none", # Legend will be placed at bottom
    aspect.ratio = panel_aspect_ratio
  )

# ---- Plot B ----

# Ensure study type has the same factor levels as plot a)
data_2b$studyType <- factor(
  data_2b$studyType,
  levels = studytype_levels
)

# Define bins and labels (show all except 0.3-0.4)
x_labels_b <- sort(unique(data_2b$mafBinRange))
x_labels_b[x_labels_b == "0.3-0.4"] <- ""
x_breaks_b <- sort(unique(data_2b$mafBinMidpoint))

# Calculate y-axis limits to ensure all data is visible in plot b)
max_y <- max(data_2b$alteringProportionInBucketCIUpper)
y_upper <- ceiling(max_y * 10) / 10 # Round up to nearest 0.1

# Create plot b)
plot_b <- ggplot(
  data_2b,
  aes(
    x = mafBinMidpoint,
    y = alteringProportionInBucket,
    color = studyType,
    fill = studyType,
    group = studyType
  )
) +
  geom_ribbon(
    aes(
      ymin = alteringProportionInBucketCILower,
      ymax = alteringProportionInBucketCIUpper
    ),
    alpha = 0.12,
    colour = NA,
    na.rm = TRUE
  ) +
  geom_line(linewidth = 0.3, na.rm = TRUE) +
  scale_color_manual(
    values = colors,
    breaks = names(colors),
    labels = c("cis-pQTL", "eQTL", "GWAS (disease)", "GWAS (measurement)"),
    name = "studyType"
  ) +
  scale_fill_manual(
    values = colors,
    breaks = names(colors),
    labels = c("cis-pQTL", "eQTL", "GWAS (disease)", "GWAS (measurement)"),
    name = "studyType"
  ) +
  scale_x_log10(
    breaks = x_breaks_b,
    labels = x_labels_b,
    expand = c(0, 0)
  ) +
  labs(
    x = expression(log[10](mean~MAF)),
    y = "Proportion of PAV"
  ) +
  base_theme +
  coord_cartesian(ylim = c(0, max(y_upper, 0.6))) +
  theme(
    plot.margin = margin(t = 5, r = 0, b = 15, l = 15),
    axis.title.x = element_text(
      size = 8,
      face = "plain",
      color = "#434343",
      margin = margin(t = 4),
      hjust = x_axis_title_hjust_b,
      vjust = x_axis_title_vjust_b
    ),
    axis.title.y = element_text(size = 8, face = "plain", color = "#434343"),
    legend.position = "none", # Legend will be placed at bottom
    aspect.ratio = panel_aspect_ratio
  )

# ---- Plot C ----

# Categorical palette for consequence categories (used in plot c)
categorical_dark_colors <- c(
  "#BC3A19",
  "#E08145",
  "#E6CA9C",
  "#9EBAA8",
  "#2F735F"
)


fill_label_order <- c(
  "protein_altering",
  "intragenic",
  "promoter",
  "enhancer",
  "intergenic"
)

# Human-readable labels for consequence categories (shared by forest plot c and barplot d)
consequence_labels <- c(
  "protein_altering" = "Protein altering",
  "intragenic" = "Intragenic",
  "promoter" = "Promoter",
  "enhancer" = "Enhancer",
  "intergenic" = "Intergenic"
)

# Format labels: show rounded integers for segments > 5%
data_c <- data_2c %>%
  mutate(
    pConsequenceLabel = ifelse(
      pConsequenceValue > 10.0,
      sprintf("%.0f", round(pConsequenceValue)),
      ""
    ),
    percentage = pConsequenceValue / 100,
    consequenceCategory = factor(
      consequenceCategory,
      levels = fill_label_order
    )
  )

plot_c <- ggplot(
  data_c,
  aes(
    x = studyType,
    y = percentage,
    fill = consequenceCategory
  )
) +
  geom_col(width = 0.7, position = "stack") +
  geom_text(
    aes(label = pConsequenceLabel),
    position = position_stack(vjust = 0.5),
    colour = "white",
    size = 8 / .pt
  ) +
  scale_y_continuous(
    expand = c(0, 0),
    labels = function(x) {
      ifelse(x == 0, "", sprintf("%.0f%%", x * 100))
    }
  ) +
  scale_x_discrete(
    labels = c(
      "cis-pqtl" = "cis-pQTL",
      "eqtl" = "eQTL",
      "gwas-disease" = "GWAS (disease)",
      "gwas-measurement" = "GWAS (measurement)"
    )
  ) +
  scale_fill_manual(
    values = categorical_dark_colors,
    labels = consequence_labels,
    name = "Consequence category"
  ) +
  labs(
    x = "",
    y = "Replicated credible sets"
  ) +
  base_theme +
  theme(
    aspect.ratio = panel_aspect_ratio,
    axis.ticks = element_blank(),
    axis.line = element_blank(),
    legend.position = "right",
    legend.text = element_text(size = 8, hjust = 0.5),
    legend.key.size = unit(0.45, "cm"),
    legend.key.height = unit(0.59, "cm"),
    legend.box.spacing = unit(0.05, "cm"),
    legend.spacing = unit(0.5, "cm"),
    axis.text.x = element_text(
      size = 8,
      margin = margin(t = 0, b = 0),
      color = "#434343",
      angle = 45,
      hjust = 0.95
    ),
    axis.text.y = element_text(size = 8, color = "#434343", margin = margin(r = -3)),
    axis.title.x = element_text(
      size = 8,
      face = "plain",
      color = "#434343",
      margin = margin(t = 10)
    ),
    panel.spacing.x = unit(0, "cm"), # Reduce spacing between panels
    plot.margin = margin(t = 5, r = 0, b = 15, l = 0)
  )

# ---- Plot D ----

data_d <- data_2d |>
  dplyr::mutate(
    consequenceCategory = factor(
      consequenceCategory,
      # Reverse order as we flip the plot
      levels = rev(fill_label_order)
    ),
    studyType = case_when(
      studyType == "gwas-measurement" ~ "measurements",
      studyType == "gwas-disease" ~ "diseases",
    )
  )

position_dodge_w <- position_dodge(width = 0.3)

plot_d <- ggplot(
  data_d,
  aes(
    x = avgMaxAbsEstimatedBeta,
    y = consequenceCategory,
    color = studyType,
    group = studyType
  )
) +
  geom_errorbar(
    aes(
      xmin = CILower,
      xmax = CIUpper
    ),
    width = 0,
    linewidth = 0.3,
    position = position_dodge_w
  ) +
  geom_point(
    position = position_dodge_w,
    size = 1
  ) +
  scale_y_discrete(labels = consequence_labels) +
  scale_color_manual(
    values = c("diseases" = "#4F97CF", "measurements" = "#245780"),
    name = "Study type"
  ) +
  guides(
    color = guide_legend(
      nrow = 2, # 2 rows
      byrow = TRUE # Fill horizontally first
    )
  ) +
  labs(
    x = expression(mean("|" * beta * "|")),
    y = ""
  ) +
  base_theme +
  theme(
    legend.position = "bottom",
    legend.justification = c(1.16, 0.5),
    legend.text = element_text(size = 8),
    legend.key.size = unit(0.5, "cm"),
    legend.direction = "horizontal",
    legend.spacing.y = unit(-2, "cm"),
    # legend.box.spacing = unit(-1, "cm"),
    legend.margin = margin(t = 0, r = 0, b = 0, l = 0),
    axis.text.x = element_text(
      size = 8,
      color = "#434343",
      angle = 45,
      # vjust = 0.8,
      # hjust = 0.5,
      margin = margin(t = 0, b = 0)
    ),
    axis.text.y = element_text(size = 8, color = "#434343"),
    axis.title.x = element_text(
      size = 8,
      face = "plain",
      color = "#434343",
      margin = margin(t = -8),
      hjust = 0.9
    ),
    axis.title.y = element_blank(),
    plot.margin = margin(t = 5, r = 5, b = 15, l = 5)
    # No aspect.ratio: width controlled by rel_widths only, height fills row
  )

# ---- Combine plots horizontally using cowplot ----
# Revision r1: plot_a (mean |beta| vs MAF) is no longer part of Figure 2; it is
# promoted to a standalone Extended Data Fig. 9 at the bottom of this script.
# Panel mapping in the r1 figure: a = plot_b, b = plot_d, c = plot_c.

# extract axis titles using get_plot_component (cowplot function)
axis_title_b <- get_plot_component(plot_b, "xlab-b", return_all = TRUE)
axis_title_c <- get_plot_component(plot_c, "xlab-b", return_all = TRUE)
axis_title_d <- get_plot_component(plot_d, "xlab-b", return_all = TRUE)


# Published widths were c(1.2, 1.2, 1.2, 2) for (plot_a, plot_b, plot_d, plot_c).
# Dropping plot_a drops its 1.2; the remaining three keep their published
# proportions so panel c (plot_c) keeps the wider slot.
rel_widths <- c(1.2, 1.2, 2) # minimal gaps between panels
spacer <- ggplot() + theme_void()


# Combine plots (panel "a" = plot b, panel "b" = plot d, panel "c" = plot c)
# Minimal margins so plot content fills each panel (no gap between label and graph)
plots_abc <- plot_grid(
  plot_b + theme(legend.position = "none", axis.title.x = element_blank(), plot.margin = margin(t = 2, r = 0, b = 15, l = 5)),
  plot_d + theme(legend.position = "none", axis.title.x = element_blank(), plot.margin = margin(t = 0, r = 15, b = 0, l = 0)),
  plot_c + theme(legend.position = "right", axis.title.x = element_blank(), plot.margin = margin(t = 2, r = 0, b = 1, l = 5), legend.text.align = 0),
  nrow = 1,
  align = "h",
  rel_widths = rel_widths,
  labels = c("a", "b", "c"),
  label_size = 8,
  # Published offsets were c(0, -0.02, -0.08, -0.04) for four slots. plot_b takes
  # the first slot (0, as plot_a had), plot_d keeps -0.08; plot_c's slot is wider
  # here than in the published grid, so a positive offset is needed to keep its
  # label the same ~0.15 in from the panel instead of drifting into the gap.
  label_x = c(0, -0.08, 0.09)
)

# Combine x-axis titles (one per panel, same slot widths as the panel grid)
plots_abc_x_axes <- plot_grid(
  axis_title_b,
  axis_title_d,
  axis_title_c,
  nrow = 1,
  align = "h",
  rel_widths = rel_widths
)

# Study-type legend. Published figure took it from plot_a and shared it across
# panels a and b (plot_a / plot_b, identical scales and labels). plot_a is no
# longer in the grid, so it is now taken from plot_b, which carries the same
# scale_color_manual/scale_fill_manual breaks and labels -> identical legend.
# Laid out vertically (one entry per row) rather than as the published single
# horizontal row: with only one panel to serve it no longer needs ~3.6 in of
# width, and a ~1.3 in vertical block fits under panel a, which frees the space
# under panel b for legend_d.
legend_ab <- get_legend(
  plot_b +
    theme(
      legend.position = "bottom",
      legend.direction = "vertical",
      legend.text = element_text(size = 8),
      legend.key.size = unit(0.3, "cm"), # Adjust key size if needed
      legend.key.width = unit(0.5, "cm"), # Adjust key width for horizontal legend
      legend.key.height = unit(0.22, "cm"),
      legend.spacing.y = unit(0, "cm"),
      legend.margin = margin(t = 0, r = 0, b = 0, l = 0)
    )
)

# plot_c keeps its own consequence-category legend inside the panel (right),
# so its slot in the legend row stays empty, as in the published figure.
legend_c <- ggplot() +
  theme_void()

legend_d <- get_legend(
  plot_d +
    theme(
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.text = element_text(size = 8),
      legend.key.size = unit(0.3, "cm"),
      legend.key.width = unit(0.5, "cm"),
      legend.spacing.y = unit(-1.5, "cm")
    )
)

# Legend row, cell widths in inches of the 8.27 in canvas. Cells equal to the
# panel slots do not centre the legends on their panels: legend_ab lands left of
# panel a's axis, and legend_d is offset to the right of its cell by its
# legend.justification = c(1.16, 0.5). The widths below are solved so each legend
# is centred on its own panel's plotted axis (panel a 0.823-2.003 in, centre
# 1.41 in; panel b 3.133-4.293 in, centre 3.71 in): a leading spacer, then the
# study-type legend, then legend_d, then the empty legend_c cell (plot_c draws
# its own consequence legend inside its panel).
legend_cells_in <- c(0.604, 1.500, 1.945, 4.221) # sums to 8.27
plot_legend_abc <- plot_grid(
  spacer,
  legend_ab,
  legend_d,
  legend_c,
  nrow = 1,
  align = "h",
  rel_widths = legend_cells_in,
  label_size = 8
)

# Overlay the x-axis titles and the legend row on the panel grid
plot_overlaid <- plots_abc +
  inset_element(
    plots_abc_x_axes,
    0,
    0.8,
    1,
    0.3,
    align_to = "full"
  )

# The vertical study-type legend is ~0.5 in tall instead of the ~0.15 in of a
# single horizontal row, so the legend row is dropped from 0.15 to 0.11 of the
# canvas height to clear the "log10(mean MAF)" axis title above it.
plot_overlaid_with_legend <- plot_overlaid +
  inset_element(
    plot_legend_abc,
    0,
    0.11,
    1,
    0.11,
    align_to = "full"
  )

ggsave(
  "figure_2_final-r1.pdf",
  plot = plot_overlaid_with_legend,
  width = 8.27,
  height = 2.5,
  dpi = 300,
  bg = "#ffffff"
)


# ---- Extended Data Fig. 9: plot_a on its own ----
# plot_a already carries its own x-axis title (labs(x = log10(mean MAF))) and its
# own colour/fill scales; in the published Figure 2 both were suppressed by the
# grid assembly (axis.title.x = element_blank(), legend.position = "none") and
# re-supplied as separate insets. Standalone it only needs the legend switched
# back on and the x-axis title centred over the panel.
plot_ed9 <- plot_a +
  theme(
    legend.position = "right",
    legend.direction = "vertical",
    legend.text = element_text(size = 8),
    legend.key.size = unit(0.3, "cm"),
    legend.key.width = unit(0.5, "cm"),
    axis.title.x = element_text(
      size = 8,
      face = "plain",
      color = "#434343",
      margin = margin(t = 4),
      hjust = 0.5,
      vjust = x_axis_title_vjust_a
    ),
    plot.margin = margin(t = 5, r = 5, b = 5, l = 10)
  )

ggsave(
  "../extended_figures/extended_figure_9-r1.pdf",
  plot = plot_ed9,
  width = 7.19, # page width of extended_figure_10.pdf (518 pt)
  height = 3.19, # page height of extended_figure_10.pdf (230 pt)
  dpi = 300,
  bg = "#ffffff"
)

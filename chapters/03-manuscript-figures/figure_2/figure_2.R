## Script to generate Figure 2 for the manuscript
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

# ---- Plot A ----

data_2a$studyType <- factor(
  data_2a$studyType,
  levels = studytype_levels
)

# X-axis breaks for plot a) (same as in the python version)
x_breaks_a <- sort(unique(data_2a$mafBinMidpoint))
x_labels_a <- sort(unique(data_2a$mafBinRange))
x_labels_a[x_labels_a == "0.01-0.05"] <- ""

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
    linewidth = 0,
    na.rm = TRUE
  ) +
  geom_line(linewidth = 0.5, na.rm = TRUE) +
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
  scale_x_continuous(
    breaks = x_breaks_a,
    labels = x_labels_a,
    expand = c(0, 0)
  ) +
  labs(
    x = "MAF bins",
    y = expression(mean("|" * hat(beta) * "|"))
  ) +
  base_theme +
  coord_cartesian(ylim = c(0, y_upper_a)) +
  theme(
    plot.margin = margin(t = 5, r = 5, b = 15, l = 5),
    axis.title.x = element_text(
      size = 8,
      face = "plain",
      color = "#434343",
      margin = margin(t = 2)
    ),
    axis.title.y = element_text(size = 8, face = "plain", color = "#434343"),
    legend.position = "none" # Legend will be placed at bottom
  )

# ---- Plot B ----

# Ensure study type has the same factor levels as plot a)
data_2b$studyType <- factor(
  data_2b$studyType,
  levels = studytype_levels
)

# Define bins and labels
x_labels_b <- sort(unique(data_2b$mafBinRange))
x_labels_b[x_labels_b == "0.01-0.05"] <- ""
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
    linewidth = 0,
    na.rm = TRUE
  ) +
  geom_line(linewidth = 0.5, na.rm = TRUE) +
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
  scale_x_continuous(
    breaks = x_breaks_b,
    labels = x_labels_b,
    expand = c(0, 0)
  ) +
  labs(
    x = "MAF bins",
    y = "Proportion of PAV"
  ) +
  base_theme +
  coord_cartesian(ylim = c(0, max(y_upper, 0.6))) +
  theme(
    plot.margin = margin(t = 5, r = 5, b = 15, l = 5),
    axis.title.x = element_text(
      size = 8,
      face = "plain",
      color = "#434343",
      margin = margin(t = 2)
    ),
    axis.title.y = element_text(size = 8, face = "plain", color = "#434343"),
    legend.position = "none" # Legend will be placed at bottom
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
    size = 3
  ) +
  scale_y_continuous(
    expand = c(0, 0),
    labels = function(x) {
      # Remove 0 label and remove % from all labels
      ifelse(x == 0, "", sprintf("%.0f", x * 100))
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
    name = "Consequence category"
  ) +
  labs(
    x = "",
    y = "%"
  ) +
  base_theme +
  theme(
    # aspect.ratio = 1.2,
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
    width = 0.3,
    position = position_dodge_w
  ) +
  geom_point(
    position = position_dodge_w,
    size = 1.5
  ) +
  scale_color_manual(
    values = c("diseases" = "#245780", "measurements" = "#2F735F"),
    name = "Study type"
  ) +
  guides(
    color = guide_legend(
      nrow = 2, # 2 rows
      byrow = TRUE # Fill horizontally first
    )
  ) +
  labs(
    x = expression(mean("|" * hat(beta) * "|")),
    y = ""
  ) +
  base_theme +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 8),
    legend.key.size = unit(0.5, "cm"),
    legend.direction = "horizontal",
    # legend.spacing.y = unit(0, "cm"),
    legend.box.spacing = unit(0.1, "cm"),
    legend.margin = margin(t = 0, r = 0, b = 0, l = 0),
    axis.text.x = element_text(
      size = 8,
      color = "#434343",
      angle = 45,
      hjust = 0.9,
      margin = margin(t = 0, b = 0)
    ),
    axis.text.y = element_blank(),
    axis.title.x = element_text(
      size = 8,
      face = "plain",
      color = "#434343",
      margin = margin(t = -8),
      hjust = 0.2
    ),
    axis.title.y = element_blank(),
    plot.margin = margin(t = 5, r = 20, b = 15, l = 0)
  )

# ---- Combine all plots horizontally using cowplot ----

# extract axis titles using get_plot_component (cowplot function)
axis_title_a <- get_plot_component(plot_a, "xlab-b", return_all = TRUE)
axis_title_b <- get_plot_component(plot_b, "xlab-b", return_all = TRUE)
axis_title_c <- get_plot_component(plot_c, "xlab-b", return_all = TRUE)
axis_title_d <- get_plot_component(plot_d, "xlab-b", return_all = TRUE)


rel_widths <- c(1, 0.1, 1, 0.1, 1.5, 0.1, 1 / 2)  # gap a-b, gap b-c, gap c-d
spacer <- ggplot() + theme_void()

# Combine plots A and B without legends
plots_abcd <- plot_grid(
  plot_a + theme(legend.position = "none", axis.title.x = element_blank()),
  spacer,
  plot_b + theme(legend.position = "none", axis.title.x = element_blank()),
  spacer,
  plot_c + theme(legend.position = "right", axis.title.x = element_blank()),
  spacer,
  plot_d + theme(legend.position = "none", axis.title.x = element_blank()),
  nrow = 1,
  align = "h",
  rel_widths = rel_widths,
  labels = c("a", "", "b", "", "c", "", "d"),
  label_size = 8,
  label_x = c(0, 0, -0.02, 0, -0.02, 0, -0.2)
)

# Combine plots A and B without legends
plots_abcd_x_axes <- plot_grid(
  axis_title_a,
  spacer,
  axis_title_b,
  spacer,
  axis_title_c,
  spacer,
  axis_title_d,
  nrow = 1,
  align = "h",
  rel_widths = rel_widths
)

# Extract legend from plot A (same legend for A and B)
legend_ab <- get_legend(
  plot_a +
    theme(
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.text = element_text(size = 8),
      legend.key.size = unit(0.3, "cm"), # Adjust key size if needed
      legend.key.width = unit(0.5, "cm") # Adjust key width for horizontal legend
    )
)

legend_c <- ggplot() +
  theme_void()

legend_d <- get_legend(
  plot_d +
    theme(
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.text = element_text(size = 8), # Half of the default size 12
      legend.key.size = unit(0.3, "cm"), # Adjust key size if needed
      legend.key.width = unit(0.5, "cm") # Adjust key width for horizontal legend
    )
)

rel_widths_ab_merged <- c(
  rel_widths[1] + rel_widths[2] + rel_widths[3],
  rel_widths[4] + rel_widths[5],
  rel_widths[6] + rel_widths[7]
)

plot_legend_abcd <- plot_grid(
  legend_ab,
  legend_c,
  legend_d,
  nrow = 1,
  align = "h",
  rel_widths = rel_widths_ab_merged,
  label_size = 8
)


plots_abcd_x_axes_with_plots <- plot_grid(
  plots_abcd,
  plots_abcd_x_axes,
  nrow = 2,
  rel_heights = c(1, 0.2)
)


# Overlay plot_b on plot_a
plot_overlaid <- plots_abcd +
  inset_element(
    plots_abcd_x_axes,
    0,
    0.8,
    1,
    0.3,
    align_to = "full"
  )

plot_overlaid_with_legend <- plot_overlaid +
  inset_element(
    plot_legend_abcd,
    0,
    0.15,
    1,
    0.15,
    align_to = "full"
  )

ggsave(
  "figure_2_new.png",
  plot = plot_overlaid_with_legend,
  width = 8.27,
  height = 2.5,
  dpi = 300,
  bg = "#ffffff"
)

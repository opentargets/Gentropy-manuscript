# Extended Data Figure 10 - the rare-variant share of cumulative discovery over time.
#
# Answers reviewer 1, minor comment 3: on the shared y-axis of Figure 1c the rare band is a sliver,
# so the reader cannot see whether its share is growing. Both panels here are on their own percent
# scale and neither shares an axis with a common-variant count.
#
#   a - rare-variant share of cumulative disease-associated genes (%)
#   b - rare-variant share of cumulative gene-disease associations (%)
#
# The rare layer is entities NOT reachable from any common-variant study - those that would not have
# been found without rare variants. It is a reachability difference between the nested tiers of
# Figure 1c, not a count of entities "first identified" through rare variants.
#
# All plotted values are read from a single pre-computed table written by
# 01_rare_discovery_over_time.ipynb, so this script performs no analysis of its own.
#
# Run (from this directory), borrowing the figures renv library:
#   R_LIBS_SITE="$(git rev-parse --show-toplevel)/chapters/03-manuscript-figures/renv/library/macos/R-4.5/aarch64-apple-darwin25.0.0" \
#     Rscript ed10_rare_variant_discovery.R

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(scales)
  library(patchwork)
})

if (getRversion() >= "2.15.1") {
  invisible(utils::globalVariables(c("metric", "panel", "year", "rare_share_pct")))
}

# Resolve script directory (works when sourced and when run via Rscript)
if (!exists("ed10_dir")) {
  .argv <- commandArgs(trailingOnly = FALSE)
  .file_arg <- .argv[startsWith(.argv, "--file=")]
  ed10_dir <- if (length(.file_arg) > 0) {
    dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
  } else {
    tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) getwd())
  }
}

repo_root <- normalizePath(file.path(ed10_dir, "..", "..", ".."))
input_csv <- file.path(repo_root, "data", "intermediate_files", "rare_discovery_over_time-r1.csv")
output_pdf <- file.path(ed10_dir, "extended_figure_10.pdf")

stopifnot(file.exists(input_csv))

# ---------------------------------------------------------------------------------------------
# Style - matches figure_1/Figure_1_b_c.R and the extended-figure family.
# ---------------------------------------------------------------------------------------------

text_size <- 8

base_theme <- theme_minimal() +
  theme(
    text = element_text(face = "plain", color = "#434343", size = text_size),
    plot.title = element_text(face = "plain", size = text_size, hjust = 0, color = "#434343",
                             margin = margin(b = 4)),
    axis.title = element_text(size = text_size, face = "plain", color = "#434343"),
    axis.title.y = element_text(size = text_size, face = "plain", color = "#434343",
                                margin = margin(r = 2), vjust = 1),
    axis.text = element_text(size = text_size, face = "plain", color = "#434343"),
    axis.text.x = element_text(size = text_size, face = "plain", margin = margin(t = -1),
                               color = "#434343"),
    axis.title.x = element_text(size = text_size, face = "plain", color = "#434343",
                                margin = margin(t = 6)),
    axis.ticks = element_line(color = "#8a8a8a", linewidth = 0.3),
    axis.ticks.length = unit(0.08, "cm"),
    axis.minor.ticks.length = rel(0.5),
    panel.background = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    axis.line = element_line(color = "#8a8a8a", linewidth = 0.3),
    legend.position = "none",
    plot.tag = element_text(face = "bold", size = text_size + 2, color = "#434343")
  )

# `rare` colour of ANCESTRY_COLORS in the ancestry reclassification notebook, i.e. the same colour
# the rare band carries in Figure 1c and Extended Data Figure 3.
color_rare <- "#FFC000"

# Bars, matching the stacked-bar idiom of Figure 1c rather than introducing a line panel.
x_breaks <- seq(2006, 2024, by = 2)
x_minor_breaks <- seq(2007, 2023, by = 2)

#' One panel: the rare share of cumulative discovery for a single metric.
build_panel <- function(df, metric_name, title, ylab, y_max) {
  d <- df %>% filter(.data$metric == metric_name)
  stopifnot(nrow(d) == 19)
  ggplot(d, aes(x = year, y = rare_share_pct)) +
    geom_col(width = 0.8, fill = color_rare) +
    scale_x_continuous(breaks = x_breaks, minor_breaks = x_minor_breaks, expand = c(0, 0),
                       guide = guide_axis(minor.ticks = TRUE)) +
    # Identical y range in both panels, so the two shares are directly comparable by eye.
    scale_y_continuous(labels = function(x) scales::number(x, accuracy = 0.5),
                       limits = c(0, y_max), breaks = scales::breaks_width(0.5),
                       expand = expansion(mult = c(0, 0.02))) +
    labs(x = "Year", y = ylab, title = title) +
    base_theme +
    coord_cartesian(xlim = c(2006 - 0.6, 2024 + 0.6))
}

# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------

shares <- suppressMessages(readr::read_csv(input_csv, show_col_types = FALSE))

stopifnot(
  setequal(unique(shares$metric), c("disease genes", "gene-disease pairs")),
  min(shares$year) == 2006, max(shares$year) == 2024,
  !any(is.na(shares$rare_share_pct))
)

# One y range for both panels, rounded up to the next half per cent.
y_max <- ceiling(max(shares$rare_share_pct) / 0.5) * 0.5

p_a <- build_panel(
  shares, "disease genes",
  "Disease-associated genes",
  "Rare-variant share of all discoveries to date (%)",
  y_max
)
p_b <- build_panel(
  shares, "gene-disease pairs",
  "Gene–disease associations",
  "Rare-variant share of all discoveries to date (%)",
  y_max
)

# ---------------------------------------------------------------------------------------------
# Assemble: side by side, panel tags a and b.
# ---------------------------------------------------------------------------------------------

combined <- (p_a | p_b) +
  plot_annotation(tag_levels = "a") &
  theme(plot.margin = margin(t = 3, r = 8, b = 3, l = 3))

ggsave(
  filename = output_pdf, plot = combined,
  width = 7.2, height = 3.2, units = "in", device = grDevices::cairo_pdf, bg = "#ffffff"
)
cat("wrote", output_pdf, "\n")

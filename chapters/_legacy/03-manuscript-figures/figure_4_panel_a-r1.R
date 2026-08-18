# Figure 4 panel a, rendered from the raw and the resolved temporal vPS tables.
#
# The full figure_4.R cannot be run: panel c reads data/figure_4/gene_pleiotropy_by_category.csv,
# which does not exist anywhere on disk and has no producer in the repository (only figure_4.R and
# plot_d.R reference it). So this script renders panel a alone -- the only panel the trait-column
# change can touch -- twice, so the two can be compared pixel for pixel.
#
# Style constants, theme, colours, scales, limits and margins are copied verbatim from
# figure_4.R lines 12-64 and 66-137. Nothing about the plot changes but the vPS input file.
#
# Run from chapters/03-manuscript-figures so renv activates:
#   Rscript figure_4_panel_a-r1.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(patchwork)
})

setwd("../..")  # renv is already loaded; figure_4.R expects the repo root
data_dir <- "data/figure_4"
out_dir  <- "chapters/03-manuscript-figures/figure_4"

# ---- Common style constants (figure_4.R lines 12-24) ----
text_size    <- 8
text_colour  <- "#434343"
axis_colour  <- "#8a8a8a"
col_gene     <- "#245780"
col_variant  <- "#2F735F"
col_cover    <- "#A01813"

base_theme <- theme_minimal() +
  theme(
    text             = element_text(face = "plain", color = text_colour, size = text_size),
    plot.title       = element_text(face = "plain", size = text_size, hjust = 0.5, color = text_colour),
    axis.title       = element_text(size = text_size, face = "plain", color = text_colour),
    axis.title.y     = element_text(size = text_size, face = "plain", color = text_colour,
                                    margin = margin(r = 4), vjust = 1),
    axis.text        = element_text(size = text_size, face = "plain", color = text_colour),
    axis.text.x      = element_text(size = text_size, face = "bold",
                                    margin = margin(t = -1), color = text_colour),
    axis.title.x     = element_text(size = text_size, face = "plain", color = text_colour,
                                    margin = margin(t = 8)),
    axis.ticks        = element_line(color = axis_colour, linewidth = 0.3),
    axis.ticks.length = unit(0.08, "cm"),
    axis.minor.ticks.length = rel(0.5),
    panel.background  = element_blank(),
    panel.grid.major  = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor  = element_blank(),
    panel.border      = element_blank(),
    axis.line         = element_line(color = axis_colour, linewidth = 0.3),
    legend.position   = "bottom",
    legend.title      = element_blank(),
    legend.text       = element_text(face = "plain", color = text_colour, size = text_size),
    strip.background  = element_blank(),
    strip.placement   = "outside",
    strip.text.y      = element_text(size = text_size, face = "plain", color = text_colour)
  )

x_breaks <- seq(2006, 2024, by = 3)
x_minor  <- setdiff(seq(2006, 2024, by = 1), x_breaks)

render_panel_a <- function(variant_file, gene_file, coverage_file) {
  gene_pleio    <- read.csv(file.path(data_dir, gene_file),     sep = "\t", row.names = 1)
  variant_pleio <- read.csv(file.path(data_dir, variant_file),  sep = "\t", row.names = 1)
  gene_coverage <- read.csv(file.path(data_dir, coverage_file), sep = "\t", row.names = 1)

  pleio_df <- bind_rows(
    gene_pleio    %>% mutate(group = "gPS"),
    variant_pleio %>% mutate(group = "vPS")
  ) %>%
    mutate(ci_lo = mean - se * 1.96, ci_hi = mean + se * 1.96,
           year  = as.integer(year))

  cover_df <- gene_coverage %>%
    mutate(group = "Variants per gene",
           ci_lo = mean - se * 1.96, ci_hi = mean + se * 1.96,
           year  = as.integer(year))

  p_top <- ggplot(pleio_df, aes(x = year, y = mean, color = group, fill = group)) +
    geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), alpha = 0.12, linewidth = 0, color = NA) +
    geom_line(linewidth = 0.8) +
    scale_color_manual(labels = c("gPS", "vPS"),
                       values = c("gPS" = col_gene, "vPS" = col_variant)) +
    scale_fill_manual(labels = c("gPS", "vPS"),
                      values = c("gPS" = col_gene, "vPS" = col_variant)) +
    scale_x_continuous(breaks = x_breaks, minor_breaks = x_minor, expand = c(0, 0),
                       guide = guide_axis(minor.ticks = TRUE)) +
    scale_y_continuous(breaks = 1:4, expand = c(0, 0)) +
    coord_cartesian(xlim = c(2006, 2024), ylim = c(1, 5)) +
    labs(x = NULL, y = "Mean pleiotropy score") +
    base_theme +
    theme(
      axis.text.x  = element_blank(),
      axis.title.x = element_blank(),
      legend.position      = c(0.02, 0.94),
      legend.justification = c(0, 1),
      legend.background    = element_rect(fill = NA, color = NA),
      plot.margin = margin(t = 5, r = 10, b = 0, l = 5)
    )

  p_bot <- ggplot(cover_df, aes(x = year, y = mean, color = group, fill = group)) +
    geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), alpha = 0.12, linewidth = 0, color = NA) +
    geom_line(linewidth = 0.8) +
    scale_color_manual(values = c("Variants per gene" = col_cover)) +
    scale_fill_manual(values  = c("Variants per gene" = col_cover)) +
    scale_x_continuous(breaks = x_breaks, minor_breaks = x_minor, expand = c(0, 0),
                       guide = guide_axis(minor.ticks = TRUE)) +
    scale_y_continuous(breaks = 1:4, expand = c(0, 0)) +
    coord_cartesian(xlim = c(2006, 2024), ylim = c(1, 5)) +
    labs(x = "Year", y = "Variants per gene") +
    base_theme +
    theme(
      axis.text.x      = element_text(size = text_size, face = "plain",
                                      margin = margin(t = -1), color = text_colour),
      axis.line.x.top  = element_blank(),
      legend.position      = c(0.02, 0.94),
      legend.justification = c(0, 1),
      legend.background    = element_rect(fill = NA, color = NA),
      plot.margin  = margin(t = 0, r = 10, b = 5, l = 5)
    )

  (p_top / p_bot) + plot_annotation(tag_levels = list(c("a", ""))) &
    theme(plot.tag = element_text(face = "bold", size = 8, color = text_colour))
}

# figure_4.R gives panel a a relative width of 0.9 out of (0.9 + 1 + 1.2) in a 12 x 4.3 in figure.
panel_width  <- 12 * 0.9 / 3.1
panel_height <- 4.3

for (variant in c("raw", "resolved")) {
  variant_file <- if (variant == "raw") "Fig4A_stats_variant_pleiotropy.csv"
                  else "Fig4A_stats_variant_pleiotropy-r1.csv"
  suffix <- if (variant == "raw") "_panel_a_raw-r1.pdf" else "_panel_a-r1.pdf"
  out <- file.path(out_dir, paste0("figure_4", suffix))
  panel <- render_panel_a(variant_file,
                          "Fig4A_stats_gene_pleiotropy.csv",
                          "Fig4A_stats_gene_coverage.csv")
  quartz(file = out, type = "pdf", width = panel_width, height = panel_height, bg = "white")
  print(panel)
  invisible(dev.off())
  message("Saved: ", out, "   (vPS from ", variant_file, ")")
}

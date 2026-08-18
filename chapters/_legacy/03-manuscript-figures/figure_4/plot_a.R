# Plot A: Pleiotropy over time – two-panel facet with shared x-axis.
# Top panel:  gPS and vPS (Number of traits)
# Bottom panel: Gene coverage by variants (Variants per Gene)

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(gtable)
  library(grid)
})

# ---- Style (mirroring Figure_1_facet.R) ----
text_size   <- 9
col_gene    <- "#245780"
col_variant <- "#2F735F"
col_cover   <- "#A01813"

base_theme <- theme_minimal() +
  theme(
    text             = element_text(face = "plain", color = "#434343", size = text_size),
    plot.title       = element_text(face = "plain", size = text_size, hjust = 0.5, color = "#434343"),
    axis.title       = element_text(size = text_size, face = "plain", color = "#434343"),
    axis.title.y     = element_text(size = text_size, face = "plain", color = "#434343", margin = margin(r = 2), vjust = 1),
    axis.text        = element_text(size = text_size, face = "plain", color = "#434343"),
    axis.text.x      = element_text(size = text_size, face = "bold", margin = margin(t = -1), color = "#434343"),
    axis.title.x     = element_text(size = text_size, face = "plain", color = "#434343", margin = margin(t = 8)),
    axis.ticks        = element_line(color = "#8a8a8a", linewidth = 0.3),
    axis.ticks.length = unit(0.08, "cm"),
    axis.minor.ticks.length = rel(0.5),
    panel.background  = element_blank(),
    panel.grid.major  = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor  = element_blank(),
    panel.border      = element_blank(),
    axis.line         = element_line(color = "#8a8a8a", linewidth = 0.3),
    legend.position   = "bottom",
    legend.title      = element_blank(),
    legend.text       = element_text(face = "plain", color = "#434343", size = text_size),
    strip.background  = element_blank(),
    strip.placement   = "outside",
    strip.text.y      = element_text(size = text_size, face = "plain", color = "#434343")
  )

# ---- 1. Read data ----
data_dir <- "data/figure_4"
if (!dir.exists(data_dir)) data_dir <- "../../data/figure_4"
if (!dir.exists(data_dir)) stop("Cannot find data/figure_4. Run from repo root or from figure_4 folder.")

gene_pleio    <- read.csv(file.path(data_dir, "Fig4A_stats_gene_pleiotropy.csv"),
                          sep = "\t", row.names = 1)
variant_pleio <- read.csv(file.path(data_dir, "Fig4A_stats_variant_pleiotropy.csv"),
                          sep = "\t", row.names = 1)
gene_coverage <- read.csv(file.path(data_dir, "Fig4A_stats_gene_coverage.csv"),
                          sep = "\t", row.names = 1)

# ---- 2. Prepare tidy data ----
pleio_df <- bind_rows(
  gene_pleio    %>% mutate(group = "gPS"),
  variant_pleio %>% mutate(group = "vPS")
) %>%
  mutate(
    ci_lo = mean - se * 1.96,
    ci_hi = mean + se * 1.96,
    year  = as.integer(year)
  )

cover_df <- gene_coverage %>%
  mutate(
    group = "Gene coverage by variants",
    ci_lo = mean - se * 1.96,
    ci_hi = mean + se * 1.96,
    year  = as.integer(year)
  )

x_breaks <- seq(2006, 2024, by = 2)
x_minor  <- seq(2007, 2023, by = 2)

# ---- 3. Top panel: Pleiotropy scores ----
p_top <- ggplot(pleio_df, aes(x = year, y = mean, color = group, fill = group)) +
  geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi),
              alpha = 0.12, linewidth = 0, color = NA) +
  geom_line(linewidth = 0.8) +
  scale_color_manual(
    labels = c("gPS", "vPS"),
    values = c("gPS" = col_gene, "vPS" = col_variant)
  ) +
  scale_fill_manual(
    labels = c("gPS", "vPS"),
    values = c("gPS" = col_gene, "vPS" = col_variant)
  ) +
  scale_x_continuous(breaks = x_breaks, minor_breaks = x_minor,
                     expand = c(0, 0),
                     guide = guide_axis(minor.ticks = TRUE)) +
  scale_y_continuous(breaks = 1:4, expand = c(0, 0)) +
  coord_cartesian(xlim = c(2006, 2024), ylim = c(1, 5)) +
  labs(x = NULL, y = "Number of traits") +
  base_theme +
  theme(
    axis.text.x  = element_blank(),
    axis.title.x = element_blank(),
    legend.position      = c(0.02, 0.94),
    legend.justification = c(0, 1),
    legend.background    = element_rect(fill = NA, color = NA),
    plot.margin = margin(t = 5, r = 10, b = 0, l = 5)
  )

# ---- 4. Bottom panel: Gene coverage ----
p_bot <- ggplot(cover_df, aes(x = year, y = mean, color = group, fill = group)) +
  geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi),
              alpha = 0.12, linewidth = 0, color = NA) +
  geom_line(linewidth = 0.8) +
  scale_color_manual(values = c("Gene coverage by variants" = col_cover)) +
  scale_fill_manual(values  = c("Gene coverage by variants" = col_cover)) +
  scale_x_continuous(breaks = x_breaks, minor_breaks = x_minor,
                     expand = c(0, 0),
                     guide = guide_axis(minor.ticks = TRUE)) +
  scale_y_continuous(breaks = 1:4, expand = c(0, 0)) +
  coord_cartesian(xlim = c(2006, 2024), ylim = c(1, 5)) +
  labs(x = "Year", y = "Variants per gene") +
  base_theme +
  theme(
    axis.text.x      = element_text(size = text_size, face = "plain", margin = margin(t = -1), color = "#434343"),
    axis.line.x.top  = element_blank(),
    legend.position      = c(0.02, 0.94),
    legend.justification = c(0, 1),
    legend.background    = element_rect(fill = NA, color = NA),
    plot.margin  = margin(t = 0, r = 10, b = 5, l = 5)
  )

# ---- 5. Stack panels with aligned widths ----
g_top <- ggplotGrob(p_top)
g_bot <- ggplotGrob(p_bot)

max_w <- unit.pmax(g_top$widths, g_bot$widths)
g_top$widths <- max_w
g_bot$widths <- max_w

rbind_g  <- getFromNamespace("rbind_gtable", "gtable")
combined <- rbind_g(g_top, g_bot, size = "max")

# ---- 6. Save ----
out_dir <- if (file.exists("chapters/03-manuscript-figures/figure_4")) {
  "chapters/03-manuscript-figures/figure_4"
} else {
  "."
}
png_file <- file.path(out_dir, "plot_a.png")
ggsave(filename = png_file, plot = combined,
       width = 3, height = 5, dpi = 300, bg = "white")
message("Saved: ", png_file)

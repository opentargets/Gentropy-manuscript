# Figure 4 combined grid: Plot A | Plot B | Plot D
# Produces a single PNG with three panels side by side.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(gtable)
  library(grid)
  library(MASS)
  library(ggplotify)
  library(patchwork)
})

# ---- Common style constants ----
text_size    <- 9
text_colour  <- "#434343"
axis_colour  <- "#8a8a8a"
col_gene     <- "#245780"
col_variant  <- "#2F735F"
col_cover    <- "#A01813"
col_uni      <- "#245780"
col_multi    <- "#528B78"
col_gwas     <- "#A01813"
col_other    <- "#245780"
col_vline    <- "#D65A1F"
axis_lwd     <- 0.8
ci_lwd       <- 0.8
pt_cex       <- 0.8
base_cex     <- 9 / 12

base_theme <- theme_minimal() +
  theme(
    text             = element_text(face = "plain", color = text_colour, size = text_size),
    plot.title       = element_text(face = "plain", size = text_size, hjust = 0.5, color = text_colour),
    axis.title       = element_text(size = text_size, face = "plain", color = text_colour),
    axis.title.y     = element_text(size = text_size, face = "plain", color = text_colour,
                                    margin = margin(r = 2), vjust = 1),
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

# ---- Data directory ----
data_dir <- "data/figure_4"
if (!dir.exists(data_dir)) data_dir <- "../../data/figure_4"
if (!dir.exists(data_dir)) data_dir <- "chapters/03-manuscript-figures/figure_4"
if (!dir.exists(data_dir)) stop("Cannot find figure_4 data directory.")

# ===========================================================================
# PLOT A: Pleiotropy over time (two-panel facet)
# ===========================================================================
gene_pleio    <- read.csv(file.path(data_dir, "Fig4A_stats_gene_pleiotropy.csv"),
                          sep = "\t", row.names = 1)
variant_pleio <- read.csv(file.path(data_dir, "Fig4A_stats_variant_pleiotropy.csv"),
                          sep = "\t", row.names = 1)
gene_coverage <- read.csv(file.path(data_dir, "Fig4A_stats_gene_coverage.csv"),
                          sep = "\t", row.names = 1)

pleio_df <- bind_rows(
  gene_pleio    %>% mutate(group = "gPS"),
  variant_pleio %>% mutate(group = "vPS")
) %>%
  mutate(ci_lo = mean - se * 1.96, ci_hi = mean + se * 1.96,
         year  = as.integer(year))

cover_df <- gene_coverage %>%
  mutate(group = "Gene coverage by variants",
         ci_lo = mean - se * 1.96, ci_hi = mean + se * 1.96,
         year  = as.integer(year))

x_breaks <- seq(2006, 2024, by = 2)
x_minor  <- seq(2007, 2023, by = 2)

p_top <- ggplot(pleio_df, aes(x = year, y = mean, color = group, fill = group)) +
  geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi),
              alpha = 0.12, linewidth = 0, color = NA) +
  geom_line(linewidth = 0.8) +
  scale_color_manual(labels = c("gPS", "vPS"),
                     values = c("gPS" = col_gene, "vPS" = col_variant)) +
  scale_fill_manual(labels = c("gPS", "vPS"),
                    values = c("gPS" = col_gene, "vPS" = col_variant)) +
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
    axis.text.x      = element_text(size = text_size, face = "plain",
                                    margin = margin(t = -1), color = text_colour),
    axis.line.x.top  = element_blank(),
    legend.position      = c(0.02, 0.94),
    legend.justification = c(0, 1),
    legend.background    = element_rect(fill = NA, color = NA),
    plot.margin  = margin(t = 0, r = 10, b = 5, l = 5)
  )

g_top <- ggplotGrob(p_top)
g_bot <- ggplotGrob(p_bot)
max_w <- unit.pmax(g_top$widths, g_bot$widths)
g_top$widths <- max_w
g_bot$widths <- max_w
rbind_g <- getFromNamespace("rbind_gtable", "gtable")
grob_a  <- rbind_g(g_top, g_bot, size = "max")

# ===========================================================================
# PLOT B: NB regression forest plot
# ===========================================================================
df_b <- read.csv(file.path(data_dir, "gene_pleiotropy_full_model.csv"),
                 stringsAsFactors = FALSE)

covariates <- c(
  "maxEQTLColocNormalised", "maxPQTLColocNormalised", "maxVEPNormalised",
  "maxEffectiveSampleSizeNormalised", "lofConstraintNormalised",
  "misConstraintNormalised", "geneLengthNormalised",
  "pathwayCountNormalised", "tissueSpecificityBinaryNormalised"
)
covariate_labels <- c(
  "eQTL colocalisation", "pQTL colocalisation", "PAV", "Sample size",
  "LoF constraint", "Missense constraint", "Gene length",
  "Pathway count", "Tissue specificity"
)

uni_coef <- uni_ci_lower <- uni_ci_upper <- numeric(length(covariates))
for (i in seq_along(covariates)) {
  fml <- as.formula(paste0("uniqueDiseases ~ ", covariates[i]))
  fit <- glm.nb(fml, data = df_b, maxit = 1000)
  uni_coef[i]     <- coef(fit)[covariates[i]]
  ci              <- confint.default(fit)[covariates[i], ]
  uni_ci_lower[i] <- ci[1]
  uni_ci_upper[i] <- ci[2]
}

fml_multi <- as.formula(
  paste("uniqueDiseases ~", paste(covariates, collapse = " + "))
)
fit_multi <- glm.nb(fml_multi, data = df_b, maxit = 1000)

multi_coef <- multi_ci_lower <- multi_ci_upper <- numeric(length(covariates))
for (i in seq_along(covariates)) {
  multi_coef[i]     <- coef(fit_multi)[covariates[i]]
  ci                <- confint.default(fit_multi)[covariates[i], ]
  multi_ci_lower[i] <- ci[1]
  multi_ci_upper[i] <- ci[2]
}

n_cov   <- length(covariates)
y_pos_b <- seq_len(n_cov) - 1L
offset  <- 0.12
x_range_b <- range(c(uni_ci_lower, uni_ci_upper, multi_ci_lower, multi_ci_upper),
                   na.rm = TRUE)
x_pad_b   <- diff(x_range_b) * 0.1

draw_plot_b <- function() {
  par(
    mar = c(4.5, 14, 1.5, 1),
    xaxs = "i", yaxs = "i",
    fg  = axis_colour, col = text_colour,
    col.axis = text_colour, col.lab = text_colour,
    cex.axis = base_cex, cex.lab = base_cex,
    family = "sans"
  )
  plot(NULL,
       xlim = c(-2, x_range_b[2] + x_pad_b),
       ylim = c(-0.6, max(y_pos_b) + 0.6),
       xlab = "Coefficient", ylab = "",
       yaxt = "n", xaxt = "n", main = "", bty = "n")
  abline(v = 0, col = col_vline, lty = 2, lwd = axis_lwd)
  arrows(uni_ci_lower, y_pos_b - offset, uni_ci_upper, y_pos_b - offset,
         length = 0.015, angle = 90, code = 3, col = col_uni, lwd = ci_lwd)
  points(uni_coef, y_pos_b - offset, pch = 19, col = col_uni, cex = pt_cex)
  arrows(multi_ci_lower, y_pos_b + offset, multi_ci_upper, y_pos_b + offset,
         length = 0.015, angle = 90, code = 3, col = col_multi, lwd = ci_lwd)
  points(multi_coef, y_pos_b + offset, pch = 19, col = col_multi, cex = pt_cex)
  axis(1, tck = -0.008, col = axis_colour, col.axis = text_colour,
       lwd = axis_lwd, cex.axis = base_cex)
  axis(2, at = par("usr")[3:4], labels = FALSE, tck = 0,
       lwd = axis_lwd, col = axis_colour)
  axis(2, at = y_pos_b, labels = covariate_labels, las = 1, tck = -0.008,
       col = NA, col.ticks = axis_colour, col.axis = text_colour,
       cex.axis = base_cex, lwd.ticks = axis_lwd)
  legend("bottomright", legend = c("Univariate", "Joint"),
         col = c(col_uni, col_multi), pch = 19, lty = 1, lwd = ci_lwd,
         bty = "n", cex = base_cex, text.col = text_colour)
}

# ===========================================================================
# PLOT D: Enrichment of Pleiotropy in Gene Sets
# ===========================================================================
results_df <- read.csv(file.path(data_dir, "gene_pleiotropy_by_category.csv"),
                       stringsAsFactors = FALSE)
results_df <- results_df[order(results_df$log_odds_ratio), ]
plot_data  <- results_df[complete.cases(results_df[, c("log_ci_lower", "log_ci_upper")]), ]
y_pos_d    <- seq_len(nrow(plot_data)) - 1L
pt_col     <- ifelse(grepl("gwas", tolower(plot_data$category)), col_gwas, col_other)

draw_plot_d <- function() {
  par(
    mar = c(4.5, 20, 1.5, 0.5),
    xaxs = "i", yaxs = "i",
    fg = axis_colour, col = text_colour,
    col.axis = text_colour, col.lab = text_colour, col.main = text_colour,
    cex.axis = base_cex, cex.lab = base_cex,
    family = "sans"
  )
  plot(plot_data$log_odds_ratio, y_pos_d, type = "n",
       xlim = c(-0.4, max(plot_data$log_ci_upper) * 1.1),
       ylim = c(-0.6, max(y_pos_d) + 0.6),
       xlab = "log(OR)", ylab = "",
       yaxt = "n", xaxt = "n", main = "", bty = "n")
  abline(v = 0, col = col_vline, lty = 2, lwd = axis_lwd)
  arrows(plot_data$log_ci_lower, y_pos_d, plot_data$log_ci_upper, y_pos_d,
         length = 0.015, angle = 90, code = 3, col = col_other, lwd = ci_lwd)
  points(plot_data$log_odds_ratio, y_pos_d, pch = 21, bg = pt_col, col = pt_col, cex = pt_cex)
  axis(1, tck = -0.008, col = axis_colour, col.axis = text_colour,
       lwd = axis_lwd, cex.axis = base_cex)
  axis(2, at = par("usr")[3:4], labels = FALSE, tck = 0,
       lwd = axis_lwd, col = axis_colour)
  axis(2, at = y_pos_d, labels = plot_data$label, las = 1, tck = -0.008,
       col = NA, col.ticks = axis_colour, col.axis = text_colour,
       cex.axis = base_cex, lwd.ticks = axis_lwd)
}

# ===========================================================================
# COMBINE: Plot A | Plot B | Plot D  (grid layout for exact alignment)
# ===========================================================================
grob_b <- as.grob(draw_plot_b)
grob_d <- as.grob(draw_plot_d)

out_dir <- if (file.exists("chapters/03-manuscript-figures/figure_4")) {
  "chapters/03-manuscript-figures/figure_4"
} else {
  "."
}
png_file <- file.path(out_dir, "figure_4_grid.png")
png(png_file, width = 16, height = 6.5, units = "in", res = 300, bg = "white")

grid.newpage()
tag_gp <- gpar(fontface = "bold", fontsize = 12, col = text_colour)

pushViewport(viewport(x = 0.5, y = 0.5, width = 0.98, height = 0.96))
lay <- grid.layout(
  nrow = 2, ncol = 3,
  heights = unit.c(unit(14, "pt"), unit(1, "null")),
  widths  = unit(c(1, 1.2, 1.6), "null")
)
pushViewport(viewport(layout = lay))

tags  <- c("a", "b", "c")
grobs <- list(grob_a, grob_b, grob_d)

for (i in seq_along(grobs)) {
  pushViewport(viewport(layout.pos.row = 1, layout.pos.col = i))
  grid.text(tags[i], x = unit(2, "pt"), y = 0.5,
            just = c("left", "center"), gp = tag_gp)
  popViewport()

  pushViewport(viewport(layout.pos.row = 2, layout.pos.col = i))
  grid.draw(grobs[[i]])
  popViewport()
}

popViewport(2)
dev.off()
message("Saved: ", png_file)

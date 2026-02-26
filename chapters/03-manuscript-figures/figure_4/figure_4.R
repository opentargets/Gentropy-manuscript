# Figure 4 combined grid: Plot A | Plot B | Plot C
# Produces a single PNG with three panels side by side.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(MASS)
  library(patchwork)
})

# ---- Common style constants ----
text_size    <- 8
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
ci_lwd       <- 0.4

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

# ===========================================================================
# PLOT B: NB regression forest plot (ggplot2)
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

forest_b <- data.frame(
  label = rep(factor(covariate_labels, levels = covariate_labels), 2),
  coef  = c(uni_coef, multi_coef),
  ci_lo = c(uni_ci_lower, multi_ci_lower),
  ci_hi = c(uni_ci_upper, multi_ci_upper),
  type  = factor(rep(c("Univariate", "Joint"), each = length(covariates)),
                 levels = c("Univariate", "Joint"))
)

p_b <- ggplot(forest_b, aes(x = coef, y = label, colour = type)) +
  geom_vline(xintercept = 0, colour = col_vline, linetype = "dashed",
             linewidth = 0.3) +
  geom_errorbar(aes(xmin = ci_lo, xmax = ci_hi), width = 0,
                linewidth = 0.4, position = position_dodge(0.45),
                orientation = "y") +
  geom_point(size = 1.5, position = position_dodge(0.45)) +
  scale_colour_manual(values = c("Univariate" = col_uni, "Joint" = col_multi)) +
  labs(x = "Coefficient", y = NULL) +
  base_theme +
  theme(
    axis.text.x = element_text(size = text_size, face = "plain", color = text_colour),
    legend.position      = c(1, 0),
    legend.justification = c(1, 0),
    legend.background    = element_rect(fill = NA, color = NA),
    legend.key.size      = unit(0.35, "cm")
  )

# ===========================================================================
# PLOT C: Enrichment of Pleiotropy in Gene Sets (ggplot2)
# ===========================================================================
results_df <- read.csv(file.path(data_dir, "gene_pleiotropy_by_category.csv"),
                       stringsAsFactors = FALSE)
results_df <- results_df[order(results_df$log_odds_ratio), ]
plot_data  <- results_df[complete.cases(results_df[, c("log_ci_lower", "log_ci_upper")]), ]

plot_data$label   <- factor(plot_data$label, levels = plot_data$label)
plot_data$is_gwas <- grepl("gwas", tolower(plot_data$category))

p_c <- ggplot(plot_data, aes(x = log_odds_ratio, y = label)) +
  geom_vline(xintercept = 0, colour = col_vline, linetype = "dashed",
             linewidth = 0.3) +
  geom_errorbar(aes(xmin = log_ci_lower, xmax = log_ci_upper), width = 0,
                colour = col_other, linewidth = 0.4, orientation = "y") +
  geom_point(aes(colour = is_gwas), shape = 19, size = 1.5) +
  scale_colour_manual(values = c("TRUE" = col_gwas, "FALSE" = col_other),
                      guide  = "none") +
  labs(x = "log(OR)", y = NULL) +
  base_theme +
  theme(
    axis.text.x = element_text(size = text_size, face = "plain", color = text_colour)
  )

# ===========================================================================
# COMBINE: (p_top / p_bot) | p_b | p_c  — all ggplot2, patchwork aligns heights
# ===========================================================================
final <- (p_top / p_bot) | p_b | p_c
final <- final +
  plot_layout(widths = c(1.3, 1, 1.2)) +
  plot_annotation(tag_levels = list(c("a", "", "b", "c"))) &
  theme(plot.tag = element_text(face = "bold", size = 8, color = text_colour))

out_dir <- if (file.exists("chapters/03-manuscript-figures/figure_4")) {
  "chapters/03-manuscript-figures/figure_4"
} else {
  "."
}
png_file <- file.path(out_dir, "figure_4_final.png")
ggsave(png_file, final, width = 12, height = 4.3, dpi = 300, bg = "white")
message("Saved: ", png_file)

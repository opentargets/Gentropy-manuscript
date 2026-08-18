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
# Data is at project root/data/figure_4. Ensure wd is project root.
if (file.exists("data/figure_4")) {
  # already at project root
} else if (file.exists("../../../data/figure_4")) {
  setwd("../../..")  # from figure_4 folder to project root
} else {
  stop("Cannot find figure_4 data directory. Run from repo root or cd to chapters/03-manuscript-figures/figure_4 first.")
}
data_dir <- "data/figure_4"

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
  mutate(group = "Variants per gene",
         ci_lo = mean - se * 1.96, ci_hi = mean + se * 1.96,
         year  = as.integer(year))

x_breaks <- seq(2006, 2024, by = 3)
x_minor  <- setdiff(seq(2006, 2024, by = 1), x_breaks)

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
  geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi),
              alpha = 0.12, linewidth = 0, color = NA) +
  geom_line(linewidth = 0.8) +
  scale_color_manual(values = c("Variants per gene" = col_cover)) +
  scale_fill_manual(values  = c("Variants per gene" = col_cover)) +
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

mean_effect_b  <- (uni_coef + multi_coef) / 2
sorted_labels_b <- covariate_labels[order(mean_effect_b)]   # ascending: lowest at bottom, highest at top

forest_b <- data.frame(
  label = rep(factor(covariate_labels, levels = sorted_labels_b), 2),
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
  labs(x = "Regression coefficient (95% CI)", y = NULL) +
  base_theme +
  theme(
    axis.text.x = element_text(size = text_size, face = "plain", color = text_colour),
    legend.position      = c(1, 0),
    legend.justification = c(1, 0),
    legend.background    = element_rect(fill = NA, color = NA),
    legend.key.size      = unit(0.35, "cm"),
    legend.title         = element_text(size = text_size, face = "plain", color = text_colour)
  )

# ===========================================================================
# PLOT C: Enrichment of Pleiotropy in Gene Sets (ggplot2)
# ===========================================================================
results_df <- read.csv(file.path(data_dir, "gene_pleiotropy_by_category.csv"),
                       stringsAsFactors = FALSE)
results_df <- results_df[order(results_df$log_odds_ratio), ]
plot_data  <- results_df[complete.cases(results_df[, c("log_ci_lower", "log_ci_upper")]), ]

col_insig <- "#9e9e9e"

# FDR from p-values (Benjamini–Hochberg); asterisk when FDR < 5%
plot_data$label <- gsub("^ChEMBL\\b", "ChEMBL approved drugs", plot_data$label)
plot_data$label <- gsub("^Withdrawn Drug\\b", "Targets of withdrawn drugs", plot_data$label)

plot_data$fdr     <- p.adjust(plot_data$p_value, method = "fdr")

# Label renaming (format: "Name |Source")
plot_data$label <- sub("^Q4 LoF constraint", "High (Q4) LoF constraint | GnomAD", plot_data$label)
plot_data$label <- sub("^Q1 LoF constraint", "Low (Q1) LoF constraint | GnomAD", plot_data$label)
plot_data$label <- sub("^ChEMBL", "Drugs / clinical candidates | ChEMBL", plot_data$label)
plot_data$label <- sub("^Mouse KO Mortality", "Mouse KO Mortality | IMPC", plot_data$label)
plot_data$label <- sub("^Trial Safety", "Safety stopped trial | OT", plot_data$label)
plot_data$label <- sub("^DD panel \\(gene2phenotype\\)", "Developmental Disease | g2p", plot_data$label)
plot_data$label <- sub("^OMIM", "Mendelian disease | OMIM", plot_data$label)
plot_data$label <- sub("^Withdrawn Drug", "Withdrawn drug | ChEMBL", plot_data$label)
plot_data$label <- sub("^Gene-based analysis", "Gene-based analysis | OT", plot_data$label)
plot_data$label <- sub("^Orphanet", "Orphan disease | Orphanet", plot_data$label)
plot_data$label <- sub("^Known safety events", "Known safety events | OT", plot_data$label)
plot_data$label <- sub("^Human Knockout", "Human Knockout | RGC-ME", plot_data$label)
plot_data$label <- sub("^Drosophila distant orthologs", "Drosophila ortholog | Ensembl", plot_data$label)
plot_data$label <- sub("^Essential Gene \\(DepMap\\)", "Cell essential | DepMap", plot_data$label)
plot_data$label <- sub("^Non-essential Gene \\(DepMap\\)", "Cell non-essential | DepMap", plot_data$label)
plot_data$label <- sub("^Cancer Driver \\(COSMIC\\)", "Cancer Driver | COSMIC", plot_data$label)
plot_data$label <- sub("^Cellular lethal \\(FUSIL\\)", "Cellular lethal | FUSIL", plot_data$label)
plot_data$label <- sub("^Developmental lethal \\(FUSIL\\)", "Developmental lethal | FUSIL", plot_data$label)
plot_data$label <- sub("^Subviable \\(FUSIL\\)", "Subviable | FUSIL", plot_data$label)
plot_data$label <- sub("^Viable with phenotype \\(FUSIL\\)", "Viable with phenotype | FUSIL", plot_data$label)
plot_data$label <- sub("^Viable with no phenotype \\(FUSIL\\)", "Viable with no phenotype | FUSIL", plot_data$label)

# Extract count and percentage separately, then clean label
plot_data$n_label   <- formatC(as.integer(sub(".*\\(([0-9]+)/[0-9.]+%\\)$", "\\1", plot_data$label)),
                               format = "d", big.mark = ",")
plot_data$pct_label <- sub(".*\\([0-9]+/([0-9.]+%)\\)$", "\\1", plot_data$label)
plot_data$label     <- sub(" \\([0-9]+/[0-9.]+%\\)$", "", plot_data$label)

plot_data$label_display <- as.character(plot_data$label)
plot_data$label     <- factor(plot_data$label_display, levels = plot_data$label_display)
plot_data$sig_label <- ifelse(plot_data$fdr < 0.05, "FDR < 5%", "FDR \u2265 5%")

x_lo   <- min(plot_data$log_ci_lower, na.rm = TRUE)
x_hi   <- max(plot_data$log_ci_upper, na.rm = TRUE)
x_span <- x_hi - x_lo
x_col1 <- x_hi + x_span * 0.22   # right edge of "Genes" column
x_col2 <- x_hi + x_span * 0.42   # right edge of "In set" column
y_hdr  <- nlevels(plot_data$label) + 1.2
x_axis_lo <- x_lo - 0.05 * x_span
x_axis_hi <- x_hi + 0.05 * x_span

p_c <- ggplot(plot_data, aes(x = log_odds_ratio, y = label, colour = sig_label)) +
  geom_vline(xintercept = 0, colour = col_vline, linetype = "dashed",
             linewidth = 0.3) +
  geom_errorbar(aes(xmin = log_ci_lower, xmax = log_ci_upper), width = 0,
                linewidth = 0.4, orientation = "y") +
  geom_point(shape = 19, size = 1.5) +
  geom_text(aes(x = x_col1, y = label, label = n_label),
            hjust = 1, size = text_size / .pt, color = text_colour,
            inherit.aes = FALSE) +
  geom_text(aes(x = x_col2, y = label, label = pct_label),
            hjust = 1, size = text_size / .pt, color = text_colour,
            inherit.aes = FALSE) +
  annotate("text", x = -Inf, y = y_hdr, label = "Gene/Target Set",
           hjust = 1.15, size = text_size / .pt, color = text_colour, fontface = "bold") +
  annotate("text", x = x_col1, y = y_hdr, label = "Genes",
           hjust = 1, size = text_size / .pt, color = text_colour, fontface = "bold") +
  annotate("text", x = x_col2, y = y_hdr, label = "In set",
           hjust = 1, size = text_size / .pt, color = text_colour, fontface = "bold") +
  scale_colour_manual(
    values = c("FDR < 5%" = col_other, "FDR \u2265 5%" = col_insig),
    breaks = c("FDR < 5%", "FDR \u2265 5%")
  ) +
  coord_cartesian(xlim = c(x_axis_lo, x_axis_hi), clip = "off") +
  labs(x = "Pleiotropy enrichment (logOR)", y = NULL) +
  base_theme +
  theme(
    axis.text.x          = element_text(size = text_size, face = "plain", color = text_colour),
    legend.position      = c(1, 0),
    legend.justification = c(1, 0),
    legend.background    = element_rect(fill = NA, color = NA),
    legend.key.size      = unit(0.35, "cm"),
    legend.margin        = margin(r = 15, b = 15),
    plot.margin          = margin(t = 10, r = 55, b = 5, l = 5)
  )

# ===========================================================================
# COMBINE: (p_top / p_bot) | p_b | p_c  — all ggplot2, patchwork aligns heights
# ===========================================================================
final <- (p_top / p_bot) | p_b | p_c
final <- final +
  plot_layout(widths = c(0.9, 1, 1.2)) +
  plot_annotation(tag_levels = list(c("a", "", "b", "c"))) &
  theme(plot.tag = element_text(face = "bold", size = 8, color = text_colour))

out_dir <- if (file.exists("chapters/03-manuscript-figures/figure_4")) {
  "chapters/03-manuscript-figures/figure_4"
} else {
  "."
}
png_file <- file.path(out_dir, "figure_4_control-r1.pdf")
quartz(file = png_file, type = "pdf", width = 12, height = 4.3, bg = "white")
print(final)
invisible(dev.off())
message("Saved: ", png_file)

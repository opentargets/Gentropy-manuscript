## Figure 5. R script to generate manuscript figure from Python plots
## Layout:
##   Left column:   Plot A (top: ORs, bottom: T-D pairs) - shared x-axis
##   Middle column: Plot B (top), Plot D (bottom)
##   Right column:  Plot C (top: Therapeutic Areas, bottom: gPS)
##
## Data: data/intermediate_files_refactor/ (temporal_drug_enrichment_full_chembl,
##       drug_enrichment_subsets_vs_full_l2g, drug_enrichment_other_resources,
##       figure_5b_contrasts, figure_5c_curves)
## Run from the repository root.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(patchwork)
  library(cowplot)
})

# ---- Paths ----
# Run from the repository root: tools/run_r.sh chapters/04-figures-main/figure_5/figure_5.R
data_dir <- "data/intermediate_files_refactor"

# ---- Common style constants (from figure_4.R) ----
text_size    <- 8
text_colour  <- "#434343"
axis_colour  <- "#8a8a8a"
col_gene     <- "#245780"
col_variant  <- "#2F735F"
col_cover    <- "#A01813"
col_vline    <- "#D65A1F"
ci_lwd       <- 0.4
line_lwd     <- 0.8
point_size_b <- 1.5

# Uniform margins for axis alignment across panels
mar_top <- margin(t = 5, r = 6, b = 2, l = 10)
mar_bot <- margin(t = 2, r = 6, b = 8, l = 10)
# Plot A: no right margin to eliminate gap with col_middle
mar_a_top <- margin(t = 5, r = 0, b = 2, l = 10)
mar_a_bot <- margin(t = 2, r = 10, b = 8, l = 10)

base_theme <- theme_minimal() +
  theme(
    text             = element_text(face = "plain", color = text_colour, size = text_size),
    plot.title       = element_text(face = "plain", size = text_size, hjust = 0.5, color = text_colour),
    axis.title       = element_text(size = text_size, face = "plain", color = text_colour),
    axis.title.y     = element_text(size = text_size, face = "plain", color = text_colour,
                                    margin = margin(r = 4), vjust = 1),
    axis.text        = element_text(size = text_size, face = "plain", color = text_colour),
    axis.text.x      = element_text(size = text_size, face = "plain",
                                    margin = margin(t = -1), color = text_colour),
    axis.title.x     = element_text(size = text_size, face = "plain", color = text_colour,
                                    margin = margin(t = 8)),
    axis.ticks       = element_line(color = axis_colour, linewidth = 0.3),
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
    legend.text       = element_text(face = "plain", color = text_colour, size = text_size)
  )

# =============================================================================
# PLOT A: Temporal dynamics (top: ORs, bottom: T-D pairs) - Left column
# =============================================================================
temporal <- read_csv(file.path(data_dir, "temporal_drug_enrichment_full_chembl.csv"),
                     show_col_types = FALSE)

# The reference line is the all-GWAS enrichment from the panel b table, read rather than pasted
# in, so it follows the data if the upstream definitions change.
enrich_file <- file.path(data_dir, "drug_enrichment_subsets_vs_full_l2g.csv")
all_enrich <- read_csv(enrich_file, show_col_types = FALSE)
overall_or <- all_enrich %>%
  filter(clinicalPhase == "4+", datasource == "full_l2g", drugsource == "full_chembl") %>%
  pull(odds_ratio)
stopifnot(length(overall_or) == 1)
temporal_4 <- temporal %>%
  filter(clinicalPhase == "4+") %>%
  mutate(
    datasource_num = as.numeric(datasource),
    yes_evid_high_clinphase = `yes_evid-high_clinphase`
  ) %>%
  arrange(datasource_num)

# Crop to 2010-2024
temporal_4_crop <- temporal_4 %>% filter(datasource_num >= 2010, datasource_num <= 2024)

# Main and minor year breaks for x-axis
x_breaks_main <- seq(2010, 2024, 2)
x_breaks_minor <- seq(2011, 2023, 2)

# Top: Odds ratios (figure_4 colors: col_gene, col_vline)
# X-axis at y=2 (bottom of plot) with ticks oriented down, no labels
p_a_top <- ggplot(temporal_4_crop, aes(x = datasource_num, y = odds_ratio)) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), fill = col_gene, alpha = 0.12) +
  geom_line(color = col_gene, linewidth = line_lwd) +
  geom_hline(data = data.frame(y = overall_or, label = "2025 enrichment"),
             aes(yintercept = y, color = label), linetype = "dashed", linewidth = ci_lwd) +
  scale_color_manual(values = c("2025 enrichment" = col_vline), name = NULL) +
  scale_x_continuous(
    breaks = x_breaks_main,
    minor_breaks = x_breaks_minor,
    limits = c(2010, 2024),
    expand = c(0, 0),
    guide = guide_axis(minor.ticks = TRUE, n.dodge = 1)
  ) +
  scale_y_continuous(limits = c(2, 12), expand = c(0, 0)) +
  labs(x = NULL, y = "GWAS clinical success (OR)") +
  base_theme +
  theme(
    axis.text.x         = element_blank(),
    axis.title.x        = element_blank(),
    axis.line.x         = element_line(color = axis_colour, linewidth = 0.3),
    axis.ticks.x        = element_line(color = axis_colour, linewidth = 0.3),
    axis.ticks.length.x = unit(0.15, "cm"),
    legend.position     = c(1, 1),
    legend.justification = c(1, 1),
    legend.background   = element_rect(fill = NA, color = NA),
    legend.key          = element_rect(fill = NA, color = NA),
    legend.key.width    = unit(0.6, "cm"),
    legend.key.height   = unit(0.3, "cm"),
    legend.text         = element_text(size = text_size - 1, color = text_colour)
  ) +
  guides(color = guide_legend(
    override.aes = list(linetype = "dashed", linewidth = ci_lwd)
  ))

# Bottom: T-D pairs (figure_4 color: col_variant)
# Same x-axis breaks and tick sizes as top plot
p_a_bot <- ggplot(temporal_4_crop, aes(x = datasource_num, y = yes_evid_high_clinphase)) +
  geom_col(fill = "#3583C0", width = 0.7) +
  scale_x_continuous(
    breaks = x_breaks_main,
    minor_breaks = x_breaks_minor,
    limits = c(2009.5, 2024.5),
    expand = c(0, 0),
    guide = guide_axis(minor.ticks = TRUE)
  ) +
  scale_y_continuous(limits = c(0, NA), expand = c(0, 0.05)) +
  labs(x = "Year", y = "Drug-supported GWAS (T-I)") +
  base_theme +
  theme(axis.ticks.length.x = unit(0.08, "cm"))

# Stack A (shared x) - use mar_a with r=0 to eliminate gap with col_middle
p_a_top <- p_a_top + theme(plot.margin = mar_a_top)
p_a_bot <- p_a_bot + theme(plot.margin = mar_a_bot)
p_a <- p_a_top / p_a_bot +
  plot_layout(heights = c(1, 1))

# =============================================================================
# PLOT B: Forest plot - gene categories with faceted category panels
# =============================================================================
# Review round 1 (R2-MJ-1) added the number-of-TAs pleiotropy strata; prefer the augmented table
# written by chapters/06-review-r1/fig5b-ta-stratum/fig5b_ta_contrast.py when it is present.
# The comparison against the other genetic-evidence resources runs over a different universe, so
# 02-analysis-main writes it to its own table. It is the "Other" facet block of panel b.
other_file <- file.path(data_dir, "drug_enrichment_other_resources.csv")
all_enrich <- bind_rows(
  all_enrich,
  read_csv(other_file, show_col_types = FALSE) %>% mutate(drugsource = "full_chembl")
)

datasource_map <- c(
  "PAV_base"              = "Without PAV",
  "PAV_subEvid"           = "With PAV",
  "rare_base"             = "Common Variants",
  "rare_subEvid"          = "Rare Variants",
  "high-gPS_base"         = "gPS<10",
  "high-gPS_subEvid"      = "gPS>=10",
  "full_l2g"              = "All GWAS",
  "Gene-based tests"              = "Gene-based",
  "The Genomics England PanelApp" = "GEL PanelApp",
  "BigEffect_base"        = "Small effect",
  "BigEffect_subEvid"     = "Large effect",
  "low-gPS-5_subEvid"     = "gPS<=5",
  "low-gPS-5_base"        = "gPS>5",
  "TA-1_subEvid"          = "TAs=1",
  "TA-6plus_subEvid"      = "TAs>=6"
)

enrich_4 <- all_enrich %>%
  filter(clinicalPhase == "4+", drugsource != "Pharmaprojects", drugsource != "no_train_chembl") %>%
  mutate(
    datasource = recode(datasource, !!!datasource_map),
    yes_evid_high_clinphase = `yes_evid-high_clinphase`
  ) %>%
  filter(
    !datasource %in% c("Not replicated CSs", "Replicated CSs", "gPS>5", "gPS<10",
                       "Large effect (|b|>1)", "Small effect (|b|<=1)",
                       "MoreBigEffect_base", "MoreBigEffect_subEvid")
  ) %>%
  mutate(
    category = case_when(
      datasource == "All GWAS"                                              ~ "All GWAS",
      datasource %in% c("gPS>=10", "gPS<=5")                               ~ "gPS",
      datasource %in% c("TAs>=6", "TAs=1")                                 ~ "TAs",
      datasource %in% c("Rare Variants", "Common Variants")                 ~ "Rare vs Common",
      datasource %in% c("Large effect", "Small effect")                      ~ "Large vs Small Effect",
      datasource %in% c("With PAV", "Without PAV")                         ~ "With vs Without PAV",
      TRUE                                                                   ~ "Other"
    )
  )

# Top-to-bottom facet order. This reproduces the published panel, where the facet variable was
# supplied as a factor in one layer and a character in another so ggplot fell back to collation
# order; the TAs group is inserted next to gPS so the two pleiotropy metrics read together.
category_order <- c("All GWAS", "gPS", "TAs", "Large vs Small Effect", "Other",
                    "Rare vs Common", "With vs Without PAV")
enrich_4$category <- factor(enrich_4$category, levels = category_order)

unique_drugsources <- unique(enrich_4$drugsource)
colors_drug <- setNames(c(col_gene, col_variant)[seq_along(unique_drugsources)], unique_drugsources)

# Desired top-to-bottom display order within each category
ds_order <- c(
  "All GWAS",
  "gPS>=10", "gPS<=5",
  "TAs>=6", "TAs=1",
  "Rare Variants", "Common Variants",
  "Large effect", "Small effect",
  "With PAV", "Without PAV",
  "Gene-based", "ClinVar/ClinGen", "OMIM", "Orphanet",
  "GEL PanelApp", "UniProt"
)

# Panel-level significance: one star where the low-vs-high contrast survives BH at 5%. The
# contrasts and their FDRs are computed in chapters/02-analysis-main/06_therapeutic_success.ipynb;
# they used to be a fixed vector here with the P values recorded only in a comment, which would
# have gone stale silently if the upstream definitions changed.
group_to_category <- c(
  "gPS"               = "gPS",
  "therapeutic areas" = "TAs",
  "effect size"       = "Large vs Small Effect",
  "variant frequency" = "Rare vs Common",
  "PAV"               = "With vs Without PAV"
)
panel_sig_df <- read_csv(file.path(data_dir, "figure_5b_contrasts.csv"), show_col_types = FALSE) %>%
  filter(.data$fdr < 0.05) %>%
  transmute(
    category = factor(unname(group_to_category[.data$group]), levels = category_order),
    sig      = "*",
    y_mid    = 1.5   # every one of these panels has 2 rows; midpoint between positions 1 and 2
  ) %>%
  filter(!is.na(.data$category))

forest_rows <- list()
for (ds in ds_order) {
  ds_data <- filter(enrich_4, datasource == ds)
  if (nrow(ds_data) == 0) next
  for (drg in unique_drugsources) {
    drg_data <- filter(ds_data, drugsource == drg)
    for (k in seq_len(nrow(drg_data))) {
      r <- drg_data[k, ]
      lab <- gsub("_", " ", sprintf("%s (%s)", r$datasource, r$yes_evid_high_clinphase))
      forest_rows[[length(forest_rows) + 1]] <- tibble(
        odds_ratio = r$odds_ratio,
        ci_low     = r$ci_low,
        ci_high    = r$ci_high,
        label      = lab,
        drugsource = r$drugsource,
        category   = as.character(r$category)
      )
    }
  }
}
forest_df <- bind_rows(forest_rows)
# Reverse so top of ds_order displays at top of each facet panel
forest_df$label    <- factor(forest_df$label, levels = rev(unique(forest_df$label)))
forest_df$category <- factor(forest_df$category, levels = category_order)

# X axis ends at 10; truncate CIs that extend beyond
x_b_min <- 2
x_b_max <- 10
x_b_breaks <- seq(x_b_min, x_b_max, 2)
forest_df_plot <- forest_df %>%
  mutate(ci_low = pmax(ci_low, x_b_min), ci_high = pmin(ci_high, x_b_max))

p_b <- ggplot(forest_df_plot, aes(x = odds_ratio, y = label, color = drugsource)) +
  geom_errorbar(aes(xmin = ci_low, xmax = ci_high), width = 0, linewidth = ci_lwd,
                position = position_dodge(width = 0.3)) +
  geom_point(size = point_size_b, position = position_dodge(width = 0.3)) +
  geom_text(data = panel_sig_df, aes(x = x_b_max + 0.15, y = y_mid, label = sig),
            inherit.aes = FALSE, hjust = 0, vjust = 0.5,
            color = text_colour, size = (text_size + 1) / ggplot2::.pt) +
  scale_color_manual(values = colors_drug) +
  scale_x_continuous(breaks = x_b_breaks, expand = c(0, 0)) +
  facet_grid(category ~ ., scales = "free_y", space = "free_y") +
  coord_cartesian(xlim = c(x_b_min, x_b_max), clip = "off") +
  labs(x = "Odds Ratio", y = NULL) +
  base_theme +
  theme(
    axis.text.y      = element_text(size = 6, margin = margin(r = 4)),
    legend.position  = "none",
    plot.margin      = margin(t = 5, r = 18, b = 8, l = 0),
    strip.text       = element_blank(),
    strip.background = element_blank(),
    panel.border     = element_rect(color = axis_colour, fill = NA, linewidth = 0.3),
    panel.spacing    = unit(0.2, "cm"),
    axis.line        = element_blank()
  )

# =============================================================================
# PLOT C: Pleiotropy regression (Right column - TA top, gPS bottom)
# =============================================================================
run_pleio_plot <- function(curves, x_var, x_label, x_breaks = NULL, show_legend = FALSE,
                           top_panel = FALSE) {
  # The logistic fit and its 200-resample bootstrap now live in
  # chapters/02-analysis-main/06_therapeutic_success.ipynb, which writes figure_5c_curves.csv.
  # This function only draws it.
  cv <- dplyr::filter(curves, .data$pleiotropy == x_var) %>% dplyr::arrange(.data$x)
  stopifnot(nrow(cv) > 0)
  x_grid <- cv$x
  x_min  <- min(x_grid)
  x_max  <- max(x_grid)

  # Long-format for legend
  leg_df <- data.frame(
    x = rep(x_grid, 3),
    y = c(cv$model_gs1, cv$observed_gs1, cv$model_gs0),
    series = rep(c("Model (with GWAS)", "Observed (with GWAS)", "Model (no GWAS)"),
                 each = length(x_grid))
  )

  if (is.null(x_breaks)) x_breaks <- scales::breaks_log()(c(x_min, x_max))

  p <- ggplot() +
    geom_ribbon(aes(x = x_grid, ymin = cv$model_gs1_lo, ymax = cv$model_gs1_hi),
                fill = col_gene, alpha = 0.12) +
    geom_ribbon(aes(x = x_grid, ymin = cv$observed_lo, ymax = cv$observed_hi),
                fill = col_gene, alpha = 0.07) +
    geom_ribbon(aes(x = x_grid, ymin = cv$model_gs0_lo, ymax = cv$model_gs0_hi),
                fill = "gray", alpha = 0.12) +
    geom_line(data = leg_df, aes(x = x, y = y, color = series, linetype = series),
              linewidth = ifelse(leg_df$series == "Observed (with GWAS)", 0.5, line_lwd)) +
    scale_color_manual(
      values = c("Model (with GWAS)" = col_gene, "Observed (with GWAS)" = col_gene,
                 "Model (no GWAS)" = "gray"),
      guide = if (show_legend) {
        guide_legend(override.aes = list(linetype = c("solid", "dashed", "solid"),
                                         linewidth = c(line_lwd, 0.5, line_lwd)),
                     spacing = 0.1, keyheight = 0.4)
      } else "none"
    ) +
    scale_linetype_manual(values = c("Model (with GWAS)" = "solid", "Observed (with GWAS)" = "dashed",
                                     "Model (no GWAS)" = "solid"), guide = "none") +
    scale_x_log10(breaks = x_breaks, labels = as.character(x_breaks),
                  limits = c(1, x_max), expand = expansion(mult = 0, add = 0),
                  guide = guide_axis(minor.ticks = FALSE)) +
    coord_cartesian(ylim = c(0.1, NA)) +
    labs(x = x_label, y = "P(Success)") +
  base_theme +
  theme(
    plot.margin = margin(5, 6, if (top_panel) 2 else 5, 14),
    axis.ticks         = element_line(color = axis_colour, linewidth = 0.3),
    axis.ticks.x       = element_line(color = axis_colour, linewidth = 0.3),
    axis.ticks.x.top   = element_blank(),
    axis.ticks.length.x = unit(0.08, "cm"),
    legend.position      = c(0.35, 1),
    legend.justification = c(0, 1),
    legend.spacing.y    = unit(0.08, "cm"),
    legend.key.height   = unit(0.3, "cm"),
    legend.background   = element_rect(fill = NA, color = NA)
  )
  p
}

curves_c <- read_csv(file.path(data_dir, "figure_5c_curves.csv"), show_col_types = FALSE)

p_c_top <- run_pleio_plot(curves_c, "uniqueTherapeuticAreas", "Pleiotropy (Therapeutic Areas)",
                         x_breaks = c(1, 2, 5, 10, 20), show_legend = TRUE, top_panel = TRUE)
p_c_bot <- run_pleio_plot(curves_c, "uniqueDiseases", "Pleiotropy (gPS)",
                         x_breaks = c(1, 2, 5, 10, 20, 50))

p_c <- p_c_top / p_c_bot +
  plot_layout(heights = c(1, 1))

# =============================================================================
# COMBINE: 3 columns with aligned axes
# =============================================================================
# Left: p_a (stacked), Middle: p_b, Right: p_c (stacked)
# Use patchwork with aligned widths; shared margin/axis settings

col_left   <- p_a
col_middle <- p_b
col_right  <- p_c

# Middle column wider for bar plot; axes="keep" avoids aligning y-axes with OR plot
final <- (col_left | col_middle | col_right) +
  plot_layout(widths = c(0.9, 1, 1), axes = "keep") +
  plot_annotation(tag_levels = list(c("a", "", "b", "c", ""))) &
  theme(
    plot.tag = element_text(face = "bold", size = text_size, color = text_colour, vjust = 1)
  )

out_dir <- "chapters/04-figures-main/figure_5"
out_pdf <- file.path(out_dir, "figure_5.pdf")
ggsave(out_pdf, final, width = 10, height = 4.5, dpi = 300, bg = "white")
message("Saved: ", out_pdf)

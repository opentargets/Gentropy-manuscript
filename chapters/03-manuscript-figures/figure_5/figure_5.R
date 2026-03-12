## Figure 5. R script to generate manuscript figure from Python plots
## Layout:
##   Left column:   Plot A (top: ORs, bottom: T-D pairs) - shared x-axis
##   Middle column: Plot B (top), Plot D (bottom)
##   Right column:  Plot C (top: Therapeutic Areas, bottom: gPS)
##
## Data: data/figure_5/ (temporal_drug_enrichment, drug_enrichment_subsets, df_for_regression)
## Run from repo root or figure_5 directory.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(patchwork)
})

# ---- Paths ----
if (file.exists("data/figure_5")) {
  data_dir <- "data/figure_5"
} else if (file.exists("../../../data/figure_5")) {
  setwd("../../..")
  data_dir <- "data/figure_5"
} else {
  stop("Cannot find data/figure_5. Run from repo root or figure_5 directory.")
}

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
temporal_4 <- temporal %>%
  filter(clinicalPhase == "4+") %>%
  mutate(
    datasource_num = as.numeric(datasource),
    yes_evid_high_clinphase = `yes_evid-high_clinphase`
  ) %>%
  arrange(datasource_num)

# Crop to 2010-2024
temporal_4_crop <- temporal_4 %>% filter(datasource_num >= 2010, datasource_num <= 2024)

# Top: Odds ratios (figure_4 colors: col_gene, col_vline)
p_a_top <- ggplot(temporal_4_crop, aes(x = datasource_num, y = odds_ratio)) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), fill = col_gene, alpha = 0.12) +
  geom_line(color = col_gene, linewidth = line_lwd) +
  geom_hline(yintercept = 3.619, color = col_vline, linetype = "dashed", linewidth = ci_lwd) +
  scale_x_continuous(
    breaks = temporal_4_crop$datasource_num[seq(1, nrow(temporal_4_crop), by = 2)],
    expand = c(0.02, 0)
  ) +
  scale_y_continuous(limits = c(2, 16), expand = c(0, 0)) +
  labs(x = NULL, y = "Odds Ratio") +
  base_theme +
  theme(
    axis.text.x       = element_blank(),
    axis.title.x      = element_blank(),
    axis.line.x       = element_blank(),
    axis.ticks.x      = element_blank()
  )

# Bottom: T-D pairs (figure_4 color: col_variant)
p_a_bot <- ggplot(temporal_4, aes(x = datasource_num, y = yes_evid_high_clinphase)) +
  geom_line(color = col_variant, linewidth = line_lwd) +
  scale_x_continuous(
    breaks = temporal_4$datasource_num[seq(1, nrow(temporal_4), by = 2)],
    expand = c(0.02, 0)
  ) +
  labs(x = "Year", y = "T-I pairs with approved drug") +
  base_theme

# Stack A (shared x) - uniform margins for axis alignment
p_a_top <- p_a_top + theme(plot.margin = mar_top)
p_a_bot <- p_a_bot + theme(plot.margin = mar_bot)
p_a <- p_a_top / p_a_bot +
  plot_layout(heights = c(1, 1))

# =============================================================================
# PLOT B: Forest plot - gene categories (Middle top)
# =============================================================================
all_enrich <- read_csv(file.path(data_dir, "drug_enrichment_subsets_vs_full_l2g.csv"),
                       show_col_types = FALSE)

datasource_map <- c(
  "PAV_base"           = "Without PAV",
  "PAV_subEvid"        = "With PAV",
  "rare_base"          = "Common Variants",
  "rare_subEvid"       = "Rare Variants",
  "high-gPS_base"      = "gPS<10",
  "high-gPS_subEvid"   = "gPS>=10",
  "full_l2g"           = "All GWAS",
  "Gene-based tests"   = "Gene-based",
  "BigEffect_base"     = "Small effect",
  "BigEffect_subEvid"  = "Large effect",
  "low-gPS-5_subEvid"  = "gPS<=5",
  "low-gPS-5_base"     = "gPS>5"
)

enrich_4 <- all_enrich %>%
  filter(clinicalPhase == "4+", drugsource != "Pharmaprojects", drugsource != "no_train_chembl") %>%
  mutate(
    datasource = recode(datasource, !!!datasource_map),
    yes_evid_high_clinphase = `yes_evid-high_clinphase`
  ) %>%
  filter(
    !datasource %in% c("Not replicated CSs", "Replicated CSs", "gPS>5", "gPS<10",
                       "Large effect (|b|>1)", "Small effect (|b|<=1)")
  ) %>%
  arrange(desc(row_number()))

unique_datasources <- unique(enrich_4$datasource)
unique_drugsources <- unique(enrich_4$drugsource)
# figure_4 forest colors: col_uni, col_multi
colors_drug <- setNames(c(col_gene, col_variant)[seq_along(unique_drugsources)], unique_drugsources)

forest_rows <- list()
y_pos <- 0
for (ds in unique_datasources) {
  ds_data <- filter(enrich_4, datasource == ds)
  for (drg in unique_drugsources) {
    drg_data <- filter(ds_data, drugsource == drg)
    for (k in seq_len(nrow(drg_data))) {
      r <- drg_data[k, ]
      # Label: datasource (n) where n = yes_evid_high_clinphase; remove underscores
      lab <- gsub("_", " ", sprintf("%s (%s)", r$datasource, r$yes_evid_high_clinphase))
      forest_rows[[length(forest_rows) + 1]] <- tibble(
        odds_ratio = r$odds_ratio,
        ci_low = r$ci_low,
        ci_high = r$ci_high,
        y_pos = y_pos,
        label = lab,
        drugsource = r$drugsource
      )
      y_pos <- y_pos + 1
    }
  }
  y_pos <- y_pos + 0.5
}
forest_df <- bind_rows(forest_rows)
forest_df$label <- factor(forest_df$label, levels = rev(forest_df$label))

p_b <- ggplot(forest_df, aes(x = odds_ratio, y = label, color = drugsource)) +
  geom_errorbar(aes(xmin = ci_low, xmax = ci_high), width = 0, linewidth = ci_lwd,
                position = position_dodge(width = 0.3)) +
  geom_point(size = point_size_b, position = position_dodge(width = 0.3)) +
  scale_color_manual(values = colors_drug) +
  scale_x_continuous(limits = c(2, 8), breaks = 2:8, expand = c(0, 0)) +
  labs(x = "Odds Ratio", y = NULL) +
  base_theme +
  theme(
    axis.text.y  = element_text(size = 6),
    axis.title.x = element_blank(),
    # Large left margin pushes OR plot right so bar plot y-axis appears left of it
    plot.margin  = margin(t = 5, r = 6, b = 2, l = 10)
  )

# =============================================================================
# PLOT D: Transition success by pleiotropy (Middle bottom)
# =============================================================================
df_reg <- read_csv(file.path(data_dir, "df_for_enrichment_regression.csv"),
                   show_col_types = FALSE)

df_phase <- df_reg %>%
  mutate(
    pleio_bin = cut(uniqueTherapeuticAreas,
                    breaks = c(0, 1, 5, Inf),
                    labels = c("Low (1)", "Medium (2-5)", "High (6+)"))
  )

transitions <- list(
  list(start = 1, end = 2, name = "P1 -> P2"),
  list(start = 2, end = 3, name = "P2 -> P3"),
  list(start = 3, end = 4, name = "P3 -> P4")
)
labels_pleio <- c("Low (1)", "Medium (2-5)", "High (6+)")

trans_results <- list()
for (lb in labels_pleio) {
  subset_df <- filter(df_phase, pleio_bin == lb)
  for (tr in transitions) {
    at_start <- sum(subset_df$maxClinicalPhase >= tr$start)
    at_end   <- sum(subset_df$maxClinicalPhase >= tr$end)
    rate <- if (at_start > 0) at_end / at_start else 0
    ci <- if (at_start > 0) {
      prop.test(at_end, at_start, conf.level = 0.95)$conf.int
    } else {
      c(0, 0)
    }
    trans_results[[length(trans_results) + 1]] <- tibble(
      Pleiotropy = lb,
      Transition = tr$name,
      Success_Rate = rate,
      N_at_start = at_start,
      ci_low = ci[1],
      ci_high = ci[2]
    )
  }
}
trans_df <- bind_rows(trans_results)
trans_df$Transition <- factor(trans_df$Transition,
                              levels = sapply(transitions, function(x) x$name))
trans_df$Pleiotropy <- factor(trans_df$Pleiotropy, levels = labels_pleio)

# Manual bar positioning to match Python (Blues_d palette)
blues_pal <- c("Low (1)" = "#3182BD", "Medium (2-5)" = "#6BAED6", "High (6+)" = "#9ECAE1")

p_d <- ggplot(trans_df, aes(x = Transition, y = Success_Rate, fill = Pleiotropy)) +
  geom_bar(stat = "identity", position = position_dodge(0.8), width = 0.7) +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high),
                position = position_dodge(0.8), width = 0.2,
                linewidth = 0.4, color = "black") +
  scale_fill_manual(values = blues_pal) +
  coord_cartesian(ylim = c(0.2, 0.9)) +
  scale_y_continuous(breaks = seq(0.2, 0.9, 0.1)) +
  labs(x = NULL, y = "Transition Probability") +
  base_theme +
  theme(
    axis.title.x = element_blank(),
    plot.margin  = mar_bot
  )

# =============================================================================
# PLOT C: Pleiotropy regression (Right column - TA top, gPS bottom)
# =============================================================================
run_pleio_plot <- function(df_full, x_var, x_label, x_breaks = NULL) {
  df_valid <- filter(df_full, .data[[x_var]] >= 1)
  x_min <- min(df_valid[[x_var]])
  x_max <- max(df_valid[[x_var]])
  x_grid <- exp(seq(log(x_min), log(x_max), length.out = 200))

  formula_str <- paste0("outcome ~ geneticSupport + I(log(", x_var, "+1)) + I(log(", x_var, "+1)^2)")
  fit_base <- glm(as.formula(formula_str), data = df_full, family = binomial)

  pred_df_gs1 <- data.frame(geneticSupport = 1, x = x_grid)
  pred_df_gs0 <- data.frame(geneticSupport = 0, x = x_grid)
  names(pred_df_gs1)[2] <- x_var
  names(pred_df_gs0)[2] <- x_var

  set.seed(42)
  B <- 200
  logit_gs1 <- matrix(NA, B, length(x_grid))
  logit_gs0 <- matrix(NA, B, length(x_grid))
  lowess_gs1 <- matrix(NA, B, length(x_grid))

  for (i in seq_len(B)) {
    boot_idx <- sample(nrow(df_full), nrow(df_full), replace = TRUE)
    df_boot <- df_full[boot_idx, ]
    tryCatch({
      m <- glm(as.formula(formula_str), data = df_boot, family = binomial)
      logit_gs1[i, ] <- predict(m, newdata = pred_df_gs1, type = "response")
      logit_gs0[i, ] <- predict(m, newdata = pred_df_gs0, type = "response")
    }, error = function(e) NULL)

    sub <- df_boot[df_boot$geneticSupport == 1 & df_boot[[x_var]] >= 1, ]
    if (length(unique(sub[[x_var]])) > 3) {
      tryCatch({
        lw <- lowess(sub[[x_var]], sub$outcome, f = 0.3)
        # Interpolate to x_grid
        pred_lw <- approx(lw$x, lw$y, xout = x_grid, rule = 2)$y
        if (!any(is.nan(pred_lw))) lowess_gs1[i, ] <- pred_lw
      }, error = function(e) NULL)
    }
  }

  logit_m1 <- colMeans(logit_gs1, na.rm = TRUE)
  logit_m0 <- colMeans(logit_gs0, na.rm = TRUE)
  logit_ci1 <- apply(logit_gs1, 2, function(x) quantile(x, c(0.025, 0.975), na.rm = TRUE))
  logit_ci0 <- apply(logit_gs0, 2, function(x) quantile(x, c(0.025, 0.975), na.rm = TRUE))
  lowess_m1 <- colMeans(lowess_gs1, na.rm = TRUE)
  lowess_ci1 <- apply(lowess_gs1, 2, function(x) quantile(x, c(0.025, 0.975), na.rm = TRUE))

  pred_base_gs1 <- predict(fit_base, newdata = pred_df_gs1, type = "response")
  pred_base_gs0 <- predict(fit_base, newdata = pred_df_gs0, type = "response")

  rug_data <- df_full[df_full$geneticSupport == 1 & df_full[[x_var]] >= 1, ]

  if (is.null(x_breaks)) x_breaks <- scales::breaks_log()(c(x_min, x_max))

  p <- ggplot() +
    geom_ribbon(aes(x = x_grid, ymin = logit_ci1[1, ], ymax = logit_ci1[2, ]),
                fill = col_gene, alpha = 0.12) +
    geom_line(aes(x = x_grid, y = pred_base_gs1), color = col_gene, linewidth = line_lwd) +
    geom_ribbon(aes(x = x_grid, ymin = lowess_ci1[1, ], ymax = lowess_ci1[2, ]),
                fill = col_gene, alpha = 0.07) +
    geom_line(aes(x = x_grid, y = lowess_m1), color = col_gene, linewidth = 0.5, linetype = "dashed") +
    geom_ribbon(aes(x = x_grid, ymin = logit_ci0[1, ], ymax = logit_ci0[2, ]),
                fill = "gray", alpha = 0.12) +
    geom_line(aes(x = x_grid, y = pred_base_gs0), color = "gray", linewidth = line_lwd) +
    geom_rug(data = rug_data, aes(x = .data[[x_var]]), sides = "b", alpha = 0.2, color = col_gene) +
    scale_x_log10(breaks = x_breaks, labels = as.character(x_breaks)) +
    coord_cartesian(ylim = c(0.1, NA), xlim = c(x_min, x_max)) +
    labs(x = x_label, y = "P(Success)") +
  base_theme +
  theme(
    plot.margin = margin(5, 6, 5, 10)
  )
  p
}

p_c_top <- run_pleio_plot(df_reg, "uniqueTherapeuticAreas", "Pleiotropy (Therapeutic Areas)",
                         x_breaks = c(1, 2, 5, 10, 20))
p_c_bot <- run_pleio_plot(df_reg, "uniqueDiseases", "Pleiotropy (gPS)",
                         x_breaks = c(1, 2, 5, 10, 20, 50))

p_c <- p_c_top / p_c_bot +
  plot_layout(heights = c(1, 1))

# =============================================================================
# COMBINE: 3 columns with aligned axes
# =============================================================================
# Left: p_a (stacked), Middle: p_b / p_d, Right: p_c (stacked)
# Use patchwork with aligned widths; shared margin/axis settings

col_left   <- p_a
col_middle <- p_b / p_d + plot_layout(heights = c(1, 1))
col_right  <- p_c

# Middle column wider for bar plot; axes="keep" avoids aligning y-axes with OR plot
final <- (col_left | col_middle | col_right) +
  plot_layout(widths = c(1, 1, 1), axes = "keep") +
  plot_annotation(tag_levels = list(c("a", "", "b", "d", "c", ""))) &
  theme(
    plot.tag = element_text(face = "plain", size = text_size, color = text_colour),
    plot.tag.position = c(0.02, 0.98)
  )

out_dir <- if (dir.exists("chapters/03-manuscript-figures/figure_5")) {
  "chapters/03-manuscript-figures/figure_5"
} else {
  "."
}
ggsave(file.path(out_dir, "figure_5_final.pdf"), final, width = 11, height = 5, dpi = 300, bg = "white")
ggsave(file.path(out_dir, "figure_5_final.png"), final, width = 11, height = 5, dpi = 300, bg = "white")
message("Saved: ", file.path(out_dir, "figure_5_final.pdf"))

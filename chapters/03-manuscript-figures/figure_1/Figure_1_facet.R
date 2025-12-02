suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
  library(stringr)
  library(rlang)
})

if (getRversion() >= "2.15.1") {
  utils::globalVariables(c(
    "first_year", "cumulative", "nfe", "common_all", "all",
    "layer_nfe", "layer_common_other", "layer_rare", "year",
    "diseaseIds", "geneId"
  ))
}

# Input and output paths
input_csv <- "/Users/polina/genetics_gsea/scr/paper_figs/l2g_diseases_full.csv"
output_png <- "/Users/polina/Gentropy-manuscript/chapters/03-manuscript-figures/figure_1/Figure_1_facet.png"

# Theme to mimic matplotlib styling (Helvetica, bold title, light grid, no spines)
base_theme <- theme_minimal() +
  theme(
    text = element_text(face = "plain", color = "#434343"),
    plot.title = element_text(face = "plain", size = 10, hjust = 0.5, color = "#434343"),
    axis.title = element_text(size = 12, face = "plain", color = "#434343"),
    axis.text = element_text(size = 10, face = "plain", color = "#434343"),
    axis.text.x = element_text(size = 10, face = "bold", margin = margin(t = -1), color = "#434343"),
    axis.title.x = element_text(size = 12, face = "plain", color = "#434343", margin = margin(t = 8)),
    axis.ticks = element_blank(),
    # axis.ticks.length.x = grid::unit(1.5, "pt"),
    # axis.ticks.x = element_line(linewidth = 0.3),
    panel.background = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#ececec", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    axis.line = element_blank(),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(face = "plain", color = "#434343", size = 12),
    strip.background = element_blank(),
    strip.placement = "outside",
    strip.text.y = element_text(size = 10, face = "plain", color = "#434343")
  )

# Colors to match the python version
color_nfe <- "#BFDAEE"
color_common <- "#8ABADE"
color_all <- "#245780"

# Helper: cumulative per-year given a logical mask and key columns
get_cumulative <- function(df, mask, year_col, group_cols) {
  mask_quo <- enquo(mask)
  df %>%
    filter(!!mask_quo) %>%
    select(all_of(c(group_cols, year_col))) %>%
    distinct() %>%
    filter(.data[[year_col]] != 2025) %>%
    group_by(across(all_of(group_cols))) %>%
    summarise(first_year = min(.data[[year_col]]), .groups = "drop") %>%
    count(first_year, name = "count") %>%
    arrange(first_year) %>%
    mutate(cumulative = cumsum(count)) %>%
    rename(year = first_year)
}

# Read data
stopifnot(file.exists(input_csv))
raw <- suppressMessages(readr::read_csv(input_csv, show_col_types = FALSE))

# Expect columns: geneId, diseaseIds (list-like), year, nfe_common, non_nfe_common
# If diseaseIds is a pipe/semicolon/comma separated string, split to rows
if ("diseaseIds" %in% names(raw) && is.character(raw$diseaseIds)) {
  sep_guess <- ifelse(any(str_detect(raw$diseaseIds, "\\|")), "|",
    ifelse(any(str_detect(raw$diseaseIds, ";")), ";",
      ifelse(any(str_detect(raw$diseaseIds, ",")), ",", NA)
    )
  )
  if (!is.na(sep_guess)) {
    raw_pairs <- raw %>%
      mutate(diseaseIds = str_split(diseaseIds, fixed(sep_guess))) %>%
      unnest(diseaseIds)
  } else {
    raw_pairs <- raw
  }
} else {
  raw_pairs <- raw
}

# Ensure integer/numeric flags
for (col in c("nfe_common", "non_nfe_common")) {
  if (col %in% names(raw)) raw[[col]] <- as.integer(raw[[col]])
  if (col %in% names(raw_pairs)) raw_pairs[[col]] <- as.integer(raw_pairs[[col]])
}

# Build cumulative series for Genes (unique genes over time)
# 1) NFE common only
genes_cum_nfe <- get_cumulative(
  raw,
  mask = nfe_common == 1,
  year_col = "year",
  group_cols = c("geneId")
)
# 2) All common (nfe_common OR non_nfe_common)
genes_cum_common <- get_cumulative(
  raw,
  mask = (nfe_common == 1) | (non_nfe_common == 1),
  year_col = "year",
  group_cols = c("geneId")
)
# 3) All rows (any)
genes_cum_all <- get_cumulative(
  raw,
  mask = !is.na(nfe_common),
  year_col = "year",
  group_cols = c("geneId")
)

# Build cumulative series for Gene-Disease pairs (unique gene-disease pairs over time)
# 1) NFE common only
pairs_cum_nfe <- get_cumulative(
  raw_pairs,
  mask = nfe_common == 1,
  year_col = "year",
  group_cols = c("geneId", "diseaseIds")
)
# 2) All common (nfe_common OR non_nfe_common)
pairs_cum_common <- get_cumulative(
  raw_pairs,
  mask = (nfe_common == 1) | (non_nfe_common == 1),
  year_col = "year",
  group_cols = c("geneId", "diseaseIds")
)
# 3) All rows (any)
pairs_cum_all <- get_cumulative(
  raw_pairs,
  mask = !is.na(nfe_common),
  year_col = "year",
  group_cols = c("geneId", "diseaseIds")
)

# Merge and compute stacked layers avoiding double counting
# Genes
years_genes <- sort(unique(c(genes_cum_nfe$year, genes_cum_common$year, genes_cum_all$year)))
genes_df <- tibble(year = years_genes) %>%
  left_join(select(genes_cum_nfe, year, nfe = cumulative), by = "year") %>%
  left_join(select(genes_cum_common, year, common_all = cumulative), by = "year") %>%
  left_join(select(genes_cum_all, year, all = cumulative), by = "year") %>%
  mutate(across(c(nfe, common_all, all), ~ replace_na(., 0))) %>%
  mutate(
    layer_nfe = nfe,
    layer_common_other = pmax(common_all - nfe, 0),
    layer_rare = pmax(all - common_all, 0)
  ) %>%
  select(year, layer_nfe, layer_common_other, layer_rare) %>%
  mutate(
    panel = "Disease~associated~genes~(x~10^3)",
    metric = "genes"
  )

# Pairs
years_pairs <- sort(unique(c(pairs_cum_nfe$year, pairs_cum_common$year, pairs_cum_all$year)))
pairs_df <- tibble(year = years_pairs) %>%
  left_join(select(pairs_cum_nfe, year, nfe = cumulative), by = "year") %>%
  left_join(select(pairs_cum_common, year, common_all = cumulative), by = "year") %>%
  left_join(select(pairs_cum_all, year, all = cumulative), by = "year") %>%
  mutate(across(c(nfe, common_all, all), ~ replace_na(., 0))) %>%
  mutate(
    layer_nfe = nfe,
    layer_common_other = pmax(common_all - nfe, 0),
    layer_rare = pmax(all - common_all, 0)
  ) %>%
  select(year, layer_nfe, layer_common_other, layer_rare) %>%
  mutate(
    panel = "Unique~gene-disease~pairs~(x~10^3)",
    metric = "pairs"
  )

# Combine for facets
plot_df <- bind_rows(genes_df, pairs_df) %>%
  pivot_longer(
    cols = c(layer_nfe, layer_common_other, layer_rare),
    names_to = "layer", values_to = "value"
  ) %>%
  mutate(
    layer = factor(layer,
      levels = c("layer_nfe", "layer_common_other", "layer_rare"),
      labels = c("NFE common (MAF \u2265 0.01)", "All common (MAF \u2265 0.01)", "Rare variants (MAF \u2265 0.01)")
    ),
    year = as.integer(year)
  )

# Build stacked bars with shared x-axis; tick every 2 years
x_breaks <- seq(2006, 2024, by = 2)

# Helper to build a single-panel plot with legend inside (top-left)
build_plot <- function(df, ylab_expr) {
  ggplot(df, aes(x = year, y = value, fill = layer)) +
    geom_col(width = 0.8, position = position_stack(reverse = TRUE)) +
    scale_fill_manual(values = c(
      "NFE common (MAF \u2265 0.01)" = color_nfe,
      "All common (MAF \u2265 0.01)" = color_common,
      "Rare variants (MAF \u2265 0.01)" = color_all
    )) +
    scale_x_continuous(breaks = x_breaks, expand = c(0, 0)) +
    scale_y_continuous(labels = function(x) ifelse(x == 0, "", scales::number(x / 1000, accuracy = 1))) +
    labs(x = "Year", y = NULL) +
    base_theme +
    ylab(ylab_expr) +
    coord_cartesian(xlim = c(2006 - 0.4, 2024 + 0.4)) +
    theme(
      legend.position = c(0.02, 0.94),
      legend.justification = c(0, 1),
      legend.background = element_rect(fill = "white", color = NA)
    )
}

# Split data for each panel
plot_df_genes <- plot_df %>% filter(metric == "genes")
plot_df_pairs <- plot_df %>% filter(metric == "pairs")

# Build two plots with y-axis labels including (x 10^3)
p_genes <- build_plot(plot_df_genes, bquote(Disease ~ associated ~ genes ~ (x ~ 10^3))) +
  theme(
    axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(),
    plot.margin = margin(t = 0, r = 5, b = 5, l = 5)
  )
p_pairs <- build_plot(plot_df_pairs, bquote(Unique ~ gene - disease ~ pairs ~ (x ~ 10^3))) +
  theme(
    axis.text.x = element_text(size = 10, face = "plain", margin = margin(t = -1), color = "#434343"),
    plot.margin = margin(t = 0, r = 5, b = 0, l = 5),
    legend.position = "none"
  )

# Read additional data for line panels (from b.ipynb)
qd_csv <- "/Users/polina/genetics_gsea/scr/paper_figs/qd_sl_eff.csv"
qm_csv <- "/Users/polina/genetics_gsea/scr/paper_figs/qm_sl_eff.csv"
qd_df <- if (file.exists(qd_csv)) suppressMessages(readr::read_csv(qd_csv, show_col_types = FALSE)) else NULL
qm_df <- if (file.exists(qm_csv)) suppressMessages(readr::read_csv(qm_csv, show_col_types = FALSE)) else NULL

# Helper: cumulative stats per year (mean and 95% CI)
get_cumulative_stats <- function(df, value_col) {
  if (is.null(df)) {
    return(tibble(year = integer(), mean = numeric(), ci = numeric()))
  }
  df <- df %>% filter(.data$year != 2025)
  yrs <- sort(unique(df$year))
  res <- lapply(yrs, function(y) {
    sub <- df %>% filter(.data$year <= y)
    n <- nrow(sub)
    mu <- mean(sub[[value_col]], na.rm = TRUE)
    se <- if (n > 0) stats::sd(sub[[value_col]], na.rm = TRUE) / sqrt(n) else 0
    tibble(year = y, mean = mu, ci = 1.96 * se)
  })
  bind_rows(res) %>% mutate(year = as.integer(year))
}

# Build samples panel (Average N samples)
if (!is.null(qd_df) && !is.null(qm_df)) {
  stats_qd_samples <- get_cumulative_stats(qd_df, "nSamples") %>% mutate(group = "Diseases (N samples)")
  stats_qm_samples <- get_cumulative_stats(qm_df, "nSamples") %>% mutate(group = "Measurements (N samples)")
  samples_df <- bind_rows(stats_qd_samples, stats_qm_samples) %>%
    filter(year >= 2006, year <= 2024)

  build_samples_plot <- function(df) {
    ggplot(df, aes(x = year, y = mean, color = group, fill = group, linetype = group)) +
      geom_ribbon(aes(ymin = mean - ci, ymax = mean + ci), alpha = 0.12, linewidth = 0) +
      # CI boundary lines to improve visibility
      geom_line(aes(y = mean + ci, linetype = group), linewidth = 0.5, alpha = 0.6, show.legend = FALSE) +
      geom_line(aes(y = mean - ci, linetype = group), linewidth = 0.5, alpha = 0.6, show.legend = FALSE) +
      geom_line(linewidth = 0.8) +
      scale_color_manual(
        breaks = c("Diseases (N samples)", "Measurements (N samples)"),
        labels = c("Diseases", "Measurements"),
        values = c("Diseases (N samples)" = "#245780", "Measurements (N samples)" = "#2F735F")
      ) +
      scale_fill_manual(
        breaks = c("Diseases (N samples)", "Measurements (N samples)"),
        labels = c("Diseases", "Measurements"),
        values = c("Diseases (N samples)" = "#245780", "Measurements (N samples)" = "#2F735F")
      ) +
      scale_linetype_manual(
        breaks = c("Diseases (N samples)", "Measurements (N samples)"),
        labels = c("Diseases", "Measurements"),
        values = c("Diseases (N samples)" = "solid", "Measurements (N samples)" = "dashed")
      ) +
      scale_x_continuous(breaks = x_breaks, expand = c(0, 0)) +
      scale_y_continuous(labels = function(x) ifelse(x == 0, "", scales::number(x / 1000, accuracy = 1))) +
      labs(x = "Year", y = NULL) +
      base_theme +
      ylab(bquote(Average ~ sample ~ size ~ (x ~ 10^3))) +
      coord_cartesian(xlim = c(2006 - 0.4, 2024 + 0.4)) +
      theme(
        legend.position = c(0.02, 0.94),
        legend.justification = c(0, 1),
        legend.background = element_rect(fill = "white", color = NA)
      )
  }
  p_samples <- build_samples_plot(samples_df) +
    theme(
      axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(),
      plot.margin = margin(t = 0, r = 5, b = 5, l = 5)
    )
} else {
  p_samples <- NULL
}

# Build beta panel (Average |beta|)
if (!is.null(qd_df) && !is.null(qm_df)) {
  stats_qd_beta <- get_cumulative_stats(qd_df, "absEstimatedBeta") %>% mutate(group = "Diseases")
  stats_qm_beta <- get_cumulative_stats(qm_df, "absEstimatedBeta") %>% mutate(group = "Measurements")
  beta_df <- bind_rows(stats_qd_beta, stats_qm_beta) %>%
    filter(year >= 2006, year <= 2024)

  build_beta_plot <- function(df) {
    ggplot(df, aes(x = year, y = mean, color = group, fill = group, linetype = group)) +
      geom_ribbon(aes(ymin = mean - ci, ymax = mean + ci), alpha = 0.10, linewidth = 0) +
      geom_line(linewidth = 0.8) +
      scale_color_manual(values = c("Diseases" = "#245780", "Measurements" = "#2F735F")) +
      scale_fill_manual(values = c("Diseases" = "#245780", "Measurements" = "#2F735F")) +
      scale_linetype_manual(values = c("Diseases" = "solid", "Measurements" = "dashed")) +
      scale_x_continuous(breaks = x_breaks, expand = c(0, 0)) +
      labs(x = "Year", y = NULL) +
      base_theme +
      ylab(expression(Average ~ effect ~ size ~ "|" * beta * "|")) +
      coord_cartesian(xlim = c(2006 - 0.4, 2024 + 0.4)) +
      theme(
        legend.position = c(0.63, 0.94),
        legend.justification = c(0, 1),
        legend.background = element_rect(fill = "white", color = NA)
      )
  }
  p_beta <- build_beta_plot(beta_df) +
    theme(
      axis.text.x = element_text(size = 10, face = "plain", margin = margin(t = -1), color = "#434343"),
      axis.ticks.x = element_blank(), axis.title.x = element_blank(),
      plot.margin = margin(t = 0, r = 5, b = 5, l = 5),
      legend.position = "none"
    )
} else {
  p_beta <- NULL
}

# Combine vertically using gtable (no extra packages)
g1 <- ggplotGrob(p_genes)
g2 <- ggplotGrob(p_pairs)

# If new panels exist, align widths across all, then stack with new plots on top
if (!is.null(p_samples) || !is.null(p_beta)) {
  grobs_list <- list()
  if (!is.null(p_samples)) grobs_list[[length(grobs_list) + 1]] <- ggplotGrob(p_samples)
  if (!is.null(p_beta)) grobs_list[[length(grobs_list) + 1]] <- ggplotGrob(p_beta)
  grobs_list[[length(grobs_list) + 1]] <- g1
  grobs_list[[length(grobs_list) + 1]] <- g2
  # Align widths
  maxw <- grobs_list[[1]]$widths
  if (length(grobs_list) > 1) {
    for (i in 2:length(grobs_list)) maxw <- grid::unit.pmax(maxw, grobs_list[[i]]$widths)
    for (i in 1:length(grobs_list)) grobs_list[[i]]$widths <- maxw
  }
  rbind_g <- getFromNamespace("rbind_gtable", "gtable")
  # Determine panel column span so separators align with x-axis area
  panel_left <- grobs_list[[1]]$layout$l[grobs_list[[1]]$layout$name == "panel"][1]
  panel_right <- grobs_list[[1]]$layout$r[grobs_list[[1]]$layout$name == "panel"][1]
  # Build separator line grob: thin (1 pt) spanning only the panel columns
  sep_thin <- gtable::gtable(widths = grobs_list[[1]]$widths, heights = grid::unit(1, "pt"))
  sep_thin <- gtable::gtable_add_grob(
    sep_thin,
    grobs = grid::rectGrob(gp = grid::gpar(fill = "#d0d0d0", col = NA)),
    t = 1, l = panel_left, b = 1, r = panel_right
  )
  # Interleave grobs with separators between all adjacent pairs
  sequence <- list()
  for (i in seq_len(length(grobs_list))) {
    sequence[[length(sequence) + 1]] <- grobs_list[[i]]
    if (i < length(grobs_list)) sequence[[length(sequence) + 1]] <- sep_thin
  }
  # Fold into single gtable
  combined_grob <- sequence[[1]]
  if (length(sequence) > 1) {
    for (i in 2:length(sequence)) combined_grob <- rbind_g(combined_grob, sequence[[i]], size = "max")
  }
} else {
  max_width <- grid::unit.pmax(g1$widths, g2$widths)
  g1$widths <- max_width
  g2$widths <- max_width
  rbind_g <- getFromNamespace("rbind_gtable", "gtable")
  # Determine panel column span so separators align with x-axis area
  panel_left <- g1$layout$l[g1$layout$name == "panel"][1]
  panel_right <- g1$layout$r[g1$layout$name == "panel"][1]
  # Build separator line grob: thin (1 pt) spanning only the panel columns
  sep_thin <- gtable::gtable(widths = g1$widths, heights = grid::unit(1, "pt"))
  sep_thin <- gtable::gtable_add_grob(
    sep_thin,
    grobs = grid::rectGrob(gp = grid::gpar(fill = "#d0d0d0", col = NA)),
    t = 1, l = panel_left, b = 1, r = panel_right
  )
  combined_grob <- rbind_g(g1, sep_thin, g2, size = "max")
}

# Save output
ggsave(filename = output_png, plot = combined_grob, width = 4.4, height = 11, dpi = 300, bg = "#ffffff00")

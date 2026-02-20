# Plot D: Enrichment of Pleiotropy in Gene Sets
# Logistic regression: In_Category ~ uniqueDiseases; plot log OR with 95% CI.

# ---- Style ----
text_colour <- "#434343"
axis_colour <- "#8a8a8a"
col_gwas <- "#A01813"
col_other <- "#245780"
col_vline <- "#bdbdbd"
axis_lwd <- 2

# ---- 1. Read data ----
data_path <- "data/figure_4/gene_pleiotropy_by_category.csv"
if (!file.exists(data_path)) {
  data_path <- "../../data/figure_4/gene_pleiotropy_by_category.csv"
}
if (!file.exists(data_path)) {
  stop("Cannot find gene_pleiotropy_by_category.csv. Run from repo root or from figure_4 folder.")
}

results_df <- read.csv(data_path, stringsAsFactors = FALSE)
results_df <- results_df[order(results_df$log_odds_ratio), ]

# ---- 5. Forest plot ----
plot_data <- results_df[complete.cases(results_df[, c("log_ci_lower", "log_ci_upper")]), ]

y_pos <- seq_len(nrow(plot_data)) - 1L
pt_col <- ifelse(grepl("gwas", tolower(plot_data$category)), col_gwas, col_other)

out_dir <- if (file.exists("chapters/03-manuscript-figures/figure_4")) {
  "chapters/03-manuscript-figures/figure_4"
} else {
  "."
}
png_file <- file.path(out_dir, "plot_d_2.png")
png(png_file, width = 7, height = 5, units = "in", res = 300, bg = "white")
par(
  mar = c(5, 20, 0, 0.5),
  xaxs = "i", yaxs = "i",
  fg = axis_colour,
  col = text_colour,
  col.axis = text_colour,
  col.lab = text_colour,
  col.main = text_colour,
  cex.axis = 1.1,
  cex.lab = 1.1
)
plot(
  plot_data$log_odds_ratio, y_pos,
  type = "n",
  xlim = c(-0.4, max(plot_data$log_ci_upper) * 1.1),
  ylim = c(-0.6, max(y_pos) + 0.6),
  xlab = "log(OR)", ylab = "",
  yaxt = "n", xaxt = "n",
  main = "", bty = "n"
)

abline(v = 0, col = col_vline, lty = 2, lwd = axis_lwd)
arrows(
  plot_data$log_ci_lower, y_pos,
  plot_data$log_ci_upper, y_pos,
  length = 0.02, angle = 90, code = 3,
  col = col_other, lwd = axis_lwd
)
points(plot_data$log_odds_ratio, y_pos, pch = 21, bg = pt_col, col = pt_col, cex = 1.2)

axis(1, tck = -0.02, col = axis_colour, col.axis = text_colour, lwd = axis_lwd)
# Y-axis: full-length line via at spanning the entire plot range, then ticks+labels separately
axis(2, at = par("usr")[3:4], labels = FALSE, tck = 0, lwd = axis_lwd, col = axis_colour)
axis(2, at = y_pos, labels = plot_data$label, las = 1, tck = -0.02,
     col = NA, col.ticks = axis_colour, col.axis = text_colour,
     cex.axis = 1.05, lwd.ticks = axis_lwd)

dev.off()
message("Saved: ", png_file)

# Plot D: Enrichment of Pleiotropy in Gene Sets
# R equivalent of python_scr/plot_d.ipynb
# Logistic regression: In_Category ~ uniqueDiseases; plot log OR with 95% CI.
# Style from figure_3/figure_3.R (base R only, no packages required).
#
# LABELS (to match reference plot):
#   Reference shows "Category (total/pct%)" where:
#     total = number of genes in that category in the FULL gene_categories table
#     pct   = percent of those that are in the pleiotropy dataset.
#   To get the same labels, provide full category totals (see "Category totals" below).

# ---- Style (from figure_3.R) ----
text_colour <- "#434343"
grid_colour <- "#ececec"
axis_colour <- "#8a8a8a"
col_gwas <- "#A01813"
col_other <- "#245780"
col_vline <- "#bdbdbd"

# ---- 1. Paths and read data ----
# Run from repo root (data/figure_4/...) or from figure_4 (../../data/figure_4/...)
data_path <- "data/figure_4/gene_pleiotropy_by_category.csv"
if (!file.exists(data_path)) {
  data_path <- "../../data/figure_4/gene_pleiotropy_by_category.csv"
}
if (!file.exists(data_path)) {
  stop("Cannot find gene_pleiotropy_by_category.csv. Run from repo root or from figure_4 folder.")
}

df <- read.csv(data_path, stringsAsFactors = FALSE)

# ---- 2. Parse and explode 'source' (one row per gene-category pair) ----
# source is stored as string like "['Cat1', 'Cat2']"
parse_source <- function(s) {
  s <- gsub("^\\[|\\]$", "", s)
  s <- trimws(s)
  if (nchar(s) == 0L) return(character(0))
  parts <- strsplit(s, "', '", fixed = TRUE)[[1L]]
  parts <- gsub("^'|'$", "", trimws(parts))
  parts[nzchar(parts)]
}

# One row per gene (first occurrence)
df_genes <- df[!duplicated(df$geneId), c("geneId", "uniqueDiseases")]
df_genes$uniqueDiseases <- as.numeric(df_genes$uniqueDiseases)

# Long format: one row per (geneId, category)
exploded_rows <- list()
for (i in seq_len(nrow(df_genes))) {
  geneId <- df_genes$geneId[i]
  cats <- parse_source(df$source[match(geneId, df$geneId)])
  for (cat in cats) {
    if (!is.na(cat) && nzchar(cat)) {
      exploded_rows[[length(exploded_rows) + 1L]] <- data.frame(
        geneId = geneId, source = cat, stringsAsFactors = FALSE
      )
    }
  }
}
df_exploded <- do.call(rbind, exploded_rows)
if (is.null(df_exploded) || nrow(df_exploded) == 0L) {
  stop("No categories found in source column.")
}

# ---- 3. Category totals ----
# Optional: provide full totals for reference-style labels (see script header).
full_category_totals <- NULL
for (try_path in c(
  "data/figure_4/category_totals.csv",
  "../../data/figure_4/category_totals.csv",
  "data/figure_4/list_of_genes_32_categories.csv",
  "../../data/figure_4/list_of_genes_32_categories.csv",
  "data/intermediate_files/list_of_genes_32_categories.csv"
)) {
  if (!file.exists(try_path)) next
  tmp <- read.csv(try_path, stringsAsFactors = FALSE)
  if (all(c("source", "total") %in% names(tmp))) {
    full_category_totals <- setNames(as.integer(tmp$total), tmp$source)
    break
  }
  if (all(c("source", "count") %in% names(tmp))) {
    full_category_totals <- setNames(as.integer(tmp$count), tmp$source)
    break
  }
  if (any(c("targetId", "geneId") %in% names(tmp)) && "source" %in% names(tmp)) {
    tbl <- table(tmp$source)
    full_category_totals <- setNames(as.integer(tbl), names(tbl))
    break
  }
}
agg <- aggregate(geneId ~ source, data = df_exploded, FUN = function(x) length(unique(x)))
category_totals_pleio <- setNames(as.integer(agg$geneId), agg$source)
if (!is.null(full_category_totals)) {
  message("Using full category totals for reference-style labels (total/pct%).")
} else {
  message("No full category totals file found; labels show category names only.")
}

# ---- 4. Logistic regression per category: In_Category ~ uniqueDiseases ----
all_categories <- unique(df_exploded$source)
all_categories <- all_categories[!is.na(all_categories)]

results_list <- list()
for (category in all_categories) {
  genes_in_cat <- unique(df_exploded$geneId[df_exploded$source == category])

  df_genes$in_category <- as.integer(df_genes$geneId %in% genes_in_cat)
  n_in_cat <- sum(df_genes$in_category)

  fit <- tryCatch(
    glm(in_category ~ uniqueDiseases, data = df_genes, family = binomial),
    error = function(e) NULL
  )
  if (is.null(fit)) next

  if (!fit$converged) next

  b <- coef(fit)["uniqueDiseases"]
  if (is.na(b)) next

  ci <- tryCatch(confint.default(fit, "uniqueDiseases"), error = function(e) NULL)
  if (is.null(ci) || any(is.na(ci))) next

  pval <- summary(fit)$coefficients["uniqueDiseases", "Pr(>|z|)"]
  log_ci_lower <- ci[1L]
  log_ci_upper <- ci[2L]

  odds_ratio <- exp(b)
  ci_lower <- exp(log_ci_lower)
  ci_upper <- exp(log_ci_upper)

  results_list[[length(results_list) + 1L]] <- data.frame(
    category = category,
    label = category,
    odds_ratio = odds_ratio,
    log_odds_ratio = as.numeric(b),
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    log_ci_lower = log_ci_lower,
    log_ci_upper = log_ci_upper,
    p_value = pval,
    count = n_in_cat,
    stringsAsFactors = FALSE
  )
}

results_df <- do.call(rbind, results_list)
results_df <- results_df[order(results_df$log_odds_ratio), ]

# ---- 5. Plot: forest plot of log OR with 95% CI ----
plot_data <- results_df[!is.na(results_df$log_ci_lower) & !is.na(results_df$log_ci_upper), ]

y_pos <- seq_len(nrow(plot_data)) - 1L
err_left <- plot_data$log_odds_ratio - plot_data$log_ci_lower
err_right <- plot_data$log_ci_upper - plot_data$log_odds_ratio

pt_col <- ifelse(grepl("gwas", tolower(plot_data$category)), col_gwas, col_other)

# Save PNG
out_dir <- if (file.exists("chapters/03-manuscript-figures/figure_4")) {
  "chapters/03-manuscript-figures/figure_4"
} else {
  "."
}
png_file <- file.path(out_dir, "plot_d.png")
png(png_file, width = 5.5, height = 5, units = "in", res = 300, bg = "white")
par(
  mar = c(0.4, 15, 0.25, 0.2),
  xaxs = "i",
  yaxs = "i",
  fg = axis_colour,
  col = text_colour,
  col.axis = text_colour,
  col.lab = text_colour,
  col.main = text_colour,
  cex.axis = 1.1,
  cex.lab = 1.1
)
plot(
  plot_data$log_odds_ratio,
  y_pos,
  type = "n",
  xlim = c(-0.1, 0.06),
  ylim = c(0, max(y_pos) + 0.6),
  xlab = "log(OR)",
  ylab = "",
  yaxt = "n",
  xaxt = "n",
  main = "",
  bty = "n"
)
abline(v = 0, col = col_vline, lty = 2, lwd = 2)
arrows(
  plot_data$log_odds_ratio - err_left,
  y_pos,
  plot_data$log_odds_ratio + err_right,
  y_pos,
  length = 0.02,
  angle = 90,
  code = 3,
  col = col_other,
  lwd = 2
)
points(plot_data$log_odds_ratio, y_pos, pch = 21, bg = pt_col, col = pt_col, cex = 1.2)
axis(1, tck = -0.02, col = axis_colour, col.axis = text_colour, lwd = 2)
axis(2, at = y_pos, labels = plot_data$category, las = 1, tck = -0.02, col = axis_colour, col.axis = text_colour, cex.axis = 1.05, lwd = 2)
dev.off()
message("Saved: ", png_file)

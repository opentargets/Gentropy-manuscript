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

df <- read.csv(data_path, stringsAsFactors = FALSE)

# ---- 2. Parse and explode 'source' (vectorized) ----
df_genes <- df[!duplicated(df$geneId), c("geneId", "uniqueDiseases", "source")]
df_genes$uniqueDiseases <- as.numeric(df_genes$uniqueDiseases)

raw_sources <- gsub("[\\[\\]']", "", df_genes$source, perl = TRUE)
parsed <- strsplit(trimws(raw_sources), ",\\s*")
df_exploded <- data.frame(
  geneId = rep(df_genes$geneId, lengths(parsed)),
  source = trimws(unlist(parsed)),
  stringsAsFactors = FALSE
)
df_exploded <- df_exploded[nzchar(df_exploded$source), ]
df_genes$source <- NULL

if (nrow(df_exploded) == 0L) stop("No categories found in source column.")

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

# ---- 4. Logistic regression per category ----
all_categories <- unique(na.omit(df_exploded$source))

run_logistic <- function(category) {
  genes_in_cat <- unique(df_exploded$geneId[df_exploded$source == category])
  df_genes$in_category <- as.integer(df_genes$geneId %in% genes_in_cat)

  fit <- tryCatch(
    glm(in_category ~ uniqueDiseases, data = df_genes, family = binomial),
    error = function(e) NULL
  )
  if (is.null(fit) || !fit$converged) return(NULL)

  b <- coef(fit)["uniqueDiseases"]
  if (is.na(b)) return(NULL)

  ci <- tryCatch(confint.default(fit, "uniqueDiseases"), error = function(e) NULL)
  if (is.null(ci) || any(is.na(ci))) return(NULL)

  pval <- summary(fit)$coefficients["uniqueDiseases", "Pr(>|z|)"]

  data.frame(
    category = category,
    log_odds_ratio = as.numeric(b),
    log_ci_lower = ci[1L],
    log_ci_upper = ci[2L],
    odds_ratio = exp(b),
    ci_lower = exp(ci[1L]),
    ci_upper = exp(ci[2L]),
    p_value = pval,
    count = sum(df_genes$in_category),
    stringsAsFactors = FALSE
  )
}

results_df <- do.call(rbind, lapply(all_categories, run_logistic))
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
png_file <- file.path(out_dir, "plot_d.png")
png(png_file, width = 5.5, height = 5, units = "in", res = 300, bg = "white")
par(
  mar = c(5, 15, 0, 0.5),
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
  xlim = c(-0.1, 0.06),
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
axis(2, at = y_pos, labels = plot_data$category, las = 1, tck = -0.02,
     col = NA, col.ticks = axis_colour, col.axis = text_colour,
     cex.axis = 1.05, lwd.ticks = axis_lwd)

dev.off()
message("Saved: ", png_file)

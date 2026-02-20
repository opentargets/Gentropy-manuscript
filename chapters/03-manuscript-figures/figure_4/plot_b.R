# Plot B: Negative Binomial regression forest plot – Unique Diseases Full Model
# Univariate and joint (multivariate) coefficients with 95% CIs.

library(MASS)

# ---- Style ----
text_colour <- "#434343"
axis_colour <- "#8a8a8a"
col_uni     <- "#1f77b4"
col_multi   <- "#ff7f0e"
col_vline   <- "#bdbdbd"
axis_lwd    <- 2

# ---- 1. Read data ----
data_path <- "data/figure_4/gene_pleiotropy_full_model.csv"
if (!file.exists(data_path)) {
  data_path <- "../../data/figure_4/gene_pleiotropy_full_model.csv"
}
if (!file.exists(data_path)) {
  stop("Cannot find gene_pleiotropy_full_model.csv. Run from repo root or from figure_4 folder.")
}

df <- read.csv(data_path, stringsAsFactors = FALSE)

# ---- 2. Define covariates and labels ----
covariates <- c(
  "maxEQTLColocNormalised",
  "maxPQTLColocNormalised",
  "maxVEPNormalised",
  "maxEffectiveSampleSizeNormalised",
  "lofConstraintNormalised",
  "misConstraintNormalised",
  "geneLengthNormalised",
  "pathwayCountNormalised",
  "tissueSpecificityBinaryNormalised"
)

covariate_labels <- c(
  "eQTL colocalisation",
  "pQTL colocalisation",
  "PAV",
  "Sample Size",
  "LoF constraint",
  "Missense constraint",
  "Gene length",
  "Pathway count",
  "Tissue specificity"
)

# ---- 3. Univariate NB regressions ----
uni_coef     <- numeric(length(covariates))
uni_ci_lower <- numeric(length(covariates))
uni_ci_upper <- numeric(length(covariates))

for (i in seq_along(covariates)) {
  fml <- as.formula(paste0("uniqueDiseases ~ ", covariates[i]))
  fit <- glm.nb(fml, data = df, maxit = 1000)
  uni_coef[i]     <- coef(fit)[covariates[i]]
  ci              <- confint.default(fit)[covariates[i], ]
  uni_ci_lower[i] <- ci[1]
  uni_ci_upper[i] <- ci[2]
}

# ---- 4. Multivariate NB regression ----
fml_multi <- as.formula(
  paste("uniqueDiseases ~", paste(covariates, collapse = " + "))
)
fit_multi <- glm.nb(fml_multi, data = df, maxit = 1000)

multi_coef     <- numeric(length(covariates))
multi_ci_lower <- numeric(length(covariates))
multi_ci_upper <- numeric(length(covariates))

for (i in seq_along(covariates)) {
  multi_coef[i]     <- coef(fit_multi)[covariates[i]]
  ci                <- confint.default(fit_multi)[covariates[i], ]
  multi_ci_lower[i] <- ci[1]
  multi_ci_upper[i] <- ci[2]
}

# ---- 5. Forest plot ----
n_cov  <- length(covariates)
y_pos  <- seq_len(n_cov) - 1L
offset <- 0.12

x_range <- range(
  c(uni_ci_lower, uni_ci_upper, multi_ci_lower, multi_ci_upper),
  na.rm = TRUE
)
x_pad <- diff(x_range) * 0.1

out_dir <- if (file.exists("chapters/03-manuscript-figures/figure_4")) {
  "chapters/03-manuscript-figures/figure_4"
} else {
  "."
}
png_file <- file.path(out_dir, "plot_b.png")
png(png_file, width = 7, height = 5, units = "in", res = 300, bg = "white")

par(
  mar = c(5, 14, 2, 2),
  xaxs = "i", yaxs = "i",
  fg  = axis_colour,
  col = text_colour,
  col.axis = text_colour,
  col.lab  = text_colour,
  cex.axis = 0.95,
  cex.lab  = 1.0
)

plot(
  NULL,
  xlim = c(-2, x_range[2] + x_pad),
  ylim = c(-0.6, max(y_pos) + 0.6),
  xlab = "Coefficient", ylab = "",
  yaxt = "n", xaxt = "n",
  main = "", bty = "n"
)

abline(v = 0, col = col_vline, lty = 2, lwd = axis_lwd)

x_ticks <- axTicks(1)
abline(v = x_ticks, col = col_vline, lty = 3, lwd = 0.5)

# Univariate
arrows(
  uni_ci_lower, y_pos - offset,
  uni_ci_upper, y_pos - offset,
  length = 0.02, angle = 90, code = 3,
  col = col_uni, lwd = axis_lwd
)
points(uni_coef, y_pos - offset, pch = 19, col = col_uni, cex = 1.0)

# Multivariate
arrows(
  multi_ci_lower, y_pos + offset,
  multi_ci_upper, y_pos + offset,
  length = 0.02, angle = 90, code = 3,
  col = col_multi, lwd = axis_lwd
)
points(multi_coef, y_pos + offset, pch = 19, col = col_multi, cex = 1.0)

# Axes
axis(1, tck = -0.02, col = axis_colour, col.axis = text_colour, lwd = axis_lwd)
axis(2, at = par("usr")[3:4], labels = FALSE, tck = 0,
     lwd = axis_lwd, col = axis_colour)
axis(2, at = y_pos, labels = covariate_labels, las = 1, tck = -0.02,
     col = NA, col.ticks = axis_colour, col.axis = text_colour,
     cex.axis = 0.95, lwd.ticks = axis_lwd)

legend(
  "bottomright",
  legend   = c("Univariate", "Joint"),
  col      = c(col_uni, col_multi),
  pch      = 19, lty = 1, lwd = axis_lwd,
  bty      = "n", cex = 0.9,
  text.col = text_colour
)

dev.off()
message("Saved: ", png_file)

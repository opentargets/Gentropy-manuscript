# Supplementary Results 8 — the four drug-target enrichment models.
#
#   tools/run_r.sh chapters/03-analysis-supplementary/08_enrichment_bias.R
#
# Reads the model frame written by 08_enrichment_bias.ipynb and refits the models the published
# analysis used: `glm` for the two fixed-effect ones and `lme4::glmer` for the two with a random
# therapeutic-area intercept. Ported from
# ~/Projects/EGL_and_training_set/archive/gentropy_paper/R_scripts/05_enrichment.R, which is the
# only surviving implementation. Results go back to CSV for the notebook to read.

suppressMessages(library(lme4))

root <- getwd()
frame_path <- file.path(root, "data/intermediate_files_refactor/sr8_model_frame.csv")
out_path <- file.path(root, "data/intermediate_files_refactor/sr8_enrichment_models.csv")

df <- read.csv(frame_path)
df$therapeuticArea <- as.factor(df$therapeuticArea)
cat(sprintf("pairs %d | approved %d | therapeutic areas %d\n",
            nrow(df), sum(df$outcome), nlevels(df$therapeuticArea)))

row <- function(model, fit, mixed) {
  beta <- if (mixed) lme4::fixef(fit)[["geneticSupport"]] else stats::coef(fit)[["geneticSupport"]]
  variance <- if (mixed) as.numeric(lme4::VarCorr(fit)$therapeuticArea) else NA_real_
  data.frame(
    model = model,
    beta = beta,
    or = exp(beta),
    taVariance = variance,
    taSd = if (mixed) sqrt(variance) else NA_real_
  )
}

results <- rbind(
  row("genetic support alone",
      glm(outcome ~ geneticSupport, data = df, family = binomial), FALSE),
  row("with maximum sample size",
      glm(outcome ~ geneticSupport + maxNSamplesScaled, data = df, family = binomial), FALSE),
  row("random therapeutic area only",
      glmer(outcome ~ geneticSupport + (1 | therapeuticArea), data = df, family = binomial), TRUE),
  row("random therapeutic area and sample size",
      glmer(outcome ~ geneticSupport + maxNSamplesScaled + (1 | therapeuticArea),
            data = df, family = binomial), TRUE)
)

print(results, row.names = FALSE)
write.csv(results, out_path, row.names = FALSE)
cat("written:", out_path, "\n")

# R1-MJ-1(d) — `ancestry-vs-sample-size`

Does ancestry add discovery **beyond** sample size?

## The comment

> "This increase in discovery is likely primarily driven by the overall increase
> in sample sizes of these large, pan-ancestry meta-analyses (usually still
> predominantly European), and have very little to do with the mix of ancestries
> contributing to them. These different impacts can be teased out of the data by
> accounting for proportion of European ancestry and overall sample size, which
> I would encourage you to do. Otherwise, the framing needs to be changed."

Also bears on **R1-mn-16** and **R1-mn-17** ("You seem to be conflating
statistical power due to sample size and possible advantages of more diverse
cohorts, which you haven't demonstrated").

The referee offered two exits — run the analysis, or change the framing.
[`ancestry-mixed-split`](../ancestry-mixed-split/) took the framing exit; this
takes the other one. They answer different questions:

|                        | Question             | Answer                                                                                          |
| ---------------------- | -------------------- | ----------------------------------------------------------------------------------------------- |
| `ancestry-mixed-split` | **who** contributed? | genuine single-ancestry non-European studies, 23% of disease genes, against 8% for pan-ancestry |
| this notebook          | **why**?             | is it because those cohorts are ancestrally different, or just because they are big?            |

## Answer

**The referee is right about pan-ancestry meta-analyses and wrong about
genuinely non-European studies.** Holding effective sample size and publication
year fixed:

| Contrast (disease studies)                  | IRR      | 95% CI    | p          |
| ------------------------------------------- | -------- | --------- | ---------- |
| fully non-European vs fully European study  | **1.69** | 1.30–2.20 | 1.0 × 10⁻⁴ |
| `non-EUR` class vs `EUR` class              | **1.67** | 1.30–2.14 | 1.0 × 10⁻⁴ |
| `mixed` (pan-ancestry) class vs `EUR` class | **0.97** | 0.80–1.18 | 0.78       |
| per 10× effective sample size               | 3.67     | 2.75–4.89 | < 10⁻⁴     |

The `mixed` coefficient is the referee's own hypothesis, and it is confirmed:
pan-ancestry meta-analyses — which are usually still predominantly European —
discover no more than European studies of the same size. Single-ancestry
non-European studies discover **67% more novel genes** than a European study of
identical effective size and year.

Measurements behave the same way, more strongly: `non-EUR` vs `EUR` IRR **3.84**
(1.99–7.43, p = 1.0 × 10⁻⁴); `mixed` vs `EUR` 0.62 (0.31–1.27, p = 0.19). Same
conclusion for ED Fig. 3's claim.

Sample size dominates in absolute terms — a 10× larger study finds ~3.7× more
novel genes — so the referee's premise about scale is not in dispute. The point
is that ancestry survives conditioning on it.

## Design decisions

| Decision        | Choice                                                                                              | Why                                                                                                                                                                                              |
| --------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Outcome         | **novel genes per study** (first discovery), with total genes and credible-set count as sensitivity | closest to the claim under attack                                                                                                                                                                |
| Sample size     | **effective N**, `log10`                                                                            | fair comparison for binary traits with skewed case/control ratios; matches `RescaledStatistics.compute_effective_sample_size` — `prev·(1−prev)·nSamples` for binary, `nSamples` for quantitative |
| Standard errors | **clustered on cohort**, unweighted                                                                 | UKB, FinnGen, MVP and BBJ each contribute many studies                                                                                                                                           |
| Domain          | **both**, disease primary                                                                           | the disputed claim is about disease genes, but ED Fig. 3 makes the same claim for measurements                                                                                                   |

Cluster unit: `projectId` for consortium releases (e.g. `FINNGEN_R12`),
`pubmedId` otherwise, falling back to `studyId`. 1,525 clusters over 5,349
disease studies; 1,604 over 21,491 measurement studies.

**Novelty** is credited by earliest **publication date**, not year — date
resolution keeps ties rare. Where studies share the earliest date, all are
credited; this affects 1,385 disease genes and 4,169 measurement genes, and the
sensitivity outcomes (which ignore novelty entirely) show the same result.

**Sample**: studies with at least one prioritised gene in a qualifying credible
set. Studies with no credible set contribute no discovery and would be a mass of
structural zeros.

## Model

Negative binomial (NB2), consistent with the vPS/gPS modelling in Methods.
Over-dispersion is severe (variance/mean ≈ 30 for disease), so Poisson is not an
option.

```
log E[novel_genes] = β₀ + β_EUR · nfeFraction + β_N · log10(effective N) + β_year · (year − 2015)
```

Fitted in two steps because direct NB2 maximum likelihood overflows on this
data: dispersion `α` is estimated once by the Cameron–Trivedi auxiliary
regression from a Poisson fit (`α` ≈ 3.72 disease, 3.79 measurement), then held
fixed while the GLM is fitted with cluster-robust standard errors.

Four specifications:

|        | Formula                                                    |
| ------ | ---------------------------------------------------------- |
| **M1** | `nfeFraction + log10N` — the comment's literal request     |
| **M2** | `+ year_c` — **primary**                                   |
| **M3** | categorical `ancestryClass` instead of the continuous term |
| **M4** | B-spline in `nfeFraction`                                  |

### Three traps, and what was done about them

1. **Year is a confounder, and a decisive one.** "Novel" is defined against
   everything published earlier, so an identical study finds fewer novel genes
   in 2024 than in 2010, and non-European studies skew late (European studies
   were 76.9% of pre-2018 GWAS but 63.8% of 2018-onwards). Omitting year biases
   _against_ the non-European contribution — and it does: **M1 without year is
   null** (disease IRR 0.90, p = 0.29). This is not a robustness failure, it is
   the confounder doing exactly what was predicted, but it does mean the result
   must always be reported with year in the model and the reason stated.
2. **European proportion is bimodal**, not uniform: 47.8% of disease studies sit
   at exactly 1.0 and 30.1% at exactly 0.0. The categorical M3 and the spline M4
   are reported alongside the linear term for this reason. On measurements the
   point masses are severe enough (69.9% at exactly 1.0) that a `df=4` B-spline
   basis is rank deficient, so spline degrees of freedom are stepped down
   automatically until the design is full rank (disease `df=4`, measurement
   `df=3`).
3. **Studies are not independent** — handled by clustering, see above.

## Robustness

**Leave-one-cohort-out** (each of the ten largest clusters dropped in turn,
primary model refitted): the disease estimate ranges IRR 1.46–1.72 and the
measurement estimate 2.52–3.61. Every refit keeps the same sign and, with one
exception, p < 0.01.

> **The one caveat worth disclosing.** Dropping **FinnGen** entirely leaves the
> disease estimate at IRR 1.46 (95% CI 0.99–2.14, p = 0.055) — same direction
> and magnitude, but no longer conventionally significant. FinnGen is the single
> largest block of non-European disease studies, and its publication date is one
> we pinned ourselves (2024-11-04, since the release carries none), which makes
> it doubly worth stating plainly rather than leaving for the referee to find.
> The measurement result does not depend on any single cohort.

**Year specification**: the result holds whether year enters linearly (disease
IRR 1.69) or as a factor (2.01); measurements 3.37 and 2.63 respectively. It
disappears if year is omitted, for the reason given above.

**Alternative outcomes** (disease): total genes IRR 1.51 (p = 3 × 10⁻⁴),
credible sets IRR 1.55 (p < 10⁻⁴). The finding is not an artefact of how novelty
is defined.

**Raw data check**: within the top effective-sample-size quintile of disease
studies, mean novel genes per study is 8.07 for < 10% European versus 4.42 for >
90% European. The model is not inventing the pattern.

## Running it

```bash
cd chapters/06-review-r1/ancestry-vs-sample-size
uv run jupyter nbconvert --to notebook --execute --inplace 01_discovery_regression.ipynb
```

Pure pandas and statsmodels — no Spark, no Java, runs in about a minute.
[`ancestry-mixed-split/01_ancestry_reclassification.ipynb`](../ancestry-mixed-split/)
must have been run first.

Inputs: `data/25.06/output/study` (read directly with pyarrow, flat columns
only), `data/intermediate_files/study_ancestry_classification-r1.csv`,
`l2g_diseases_full-r1.csv`, `l2g_measurements_full-r1.csv`.

## Outputs

Into `data/intermediate_files/`:

| File                                     | Contents                                                                          |
| ---------------------------------------- | --------------------------------------------------------------------------------- |
| `regression_coefficients-r1.csv`         | every term of every model, both domains, on the IRR scale                         |
| `regression_headline-r1.csv`             | the `nfeFraction` coefficient, M1 and M2, both domains                            |
| `regression_model_comparison-r1.csv`     | log-likelihoods, LR χ², McFadden pseudo-R² against a scale-and-year-only model    |
| `regression_sensitivity-r1.csv`          | primary spec repeated for total genes and credible-set count                      |
| `regression_leave_one_cohort_out-r1.csv` | ten largest clusters dropped in turn                                              |
| `regression_year_specifications-r1.csv`  | year linear / factor / omitted                                                    |
| `regression_study_table-r1.csv`          | the per-study modelling frame, so the fit can be reproduced without rebuilding it |

## Where the result lands

- **Results §1**, one sentence near the new ancestry numbers.
- **The response letter**, as the direct answer to "which I would encourage you
  to do" — and it is a stronger answer than a defence, because it confirms his
  pan-ancestry hypothesis while separating it from the single-ancestry one.
- **R1-mn-17**: this supports a version of the deleted diversity claim.
  Restoring it in supported form is now an option rather than a concession.
- **Supplementary table**, full M2 output for both domains.

Both exits the referee offered are now taken, and they agree: the descriptive
split (23% vs 8%) and the conditional model (1.67 vs 0.97) tell the same story
from different directions.

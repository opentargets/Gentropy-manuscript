"""Drug-target enrichment statistics.

`or_rs` mirrors `gentropy.method.drug_enrichment_from_evid.chemblDrugEnrichment
.drug_enrichemnt_from_evidence`: Fisher's exact odds ratio, relative success as a risk ratio,
and Woolf log-scale confidence intervals with z = 1.96. `support_mask` builds the
genetic-support definition for one (PAV, pleiotropy window) combination.

Verified against the manuscript on the pair-level table: OR = 3.62 with 242 approved pairs for
all GWAS support, and OR = 10.29 / RS = 4.84 with 51 approved pairs for PAV support in 2-5
therapeutic areas. Methods "Clinical trials success modelling".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, fisher_exact, norm

Z = 1.96


def or_rs(support, approved, phase_label: str = "4+") -> dict:
    """Odds ratio and relative success for one genetic-support definition.

    Args:
        support: boolean array, genetic support present for the target-indication pair
        approved: 0/1 array, pair reached the phase of interest
        phase_label: label carried into the output row

    Returns:
        dict with the same fields the published enrichment table reports
    """
    support = np.asarray(support, dtype=bool)
    approved = np.asarray(approved, dtype=int)

    N_G = int(support.sum())
    N_negG = int((~support).sum())
    X_G = int(approved[support].sum())
    X_negG = int(approved[~support].sum())

    table = [[N_negG - X_negG, X_negG], [N_G - X_G, X_G]]
    odds_ratio, p_value = fisher_exact(table)

    row = {
        "clinicalPhase": phase_label,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "ci_low": np.nan,
        "ci_high": np.nan,
        "relative_success": np.nan,
        "ci_rs_low": np.nan,
        "ci_rs_high": np.nan,
        "rs_p_value": np.nan,
        "no_evid-low_clinphase": table[0][0],
        "no_evid-high_clinphase": X_negG,
        "yes_evid-low_clinphase": table[1][0],
        "yes_evid-high_clinphase": X_G,
        "n_support": N_G,
        "n_no_support": N_negG,
    }
    if np.any(np.array(table) == 0):
        # degenerate table: the published code returns OR = RS = 1 here
        row["odds_ratio"] = 1.0
        row["relative_success"] = 1.0
        row["rs_p_value"] = 1.0
        return row

    ln_or = np.log(odds_ratio)
    se_ln_or = np.sqrt(1 / table[0][0] + 1 / table[0][1] + 1 / table[1][0] + 1 / table[1][1])

    rs = (X_G / N_G) / (X_negG / N_negG)
    ln_rs = np.log(rs)
    se_ln_rs = np.sqrt((1 / X_negG) - (1 / N_negG) + (1 / X_G) - (1 / N_G))

    row["ci_low"] = float(np.exp(ln_or - Z * se_ln_or))
    row["ci_high"] = float(np.exp(ln_or + Z * se_ln_or))
    row["relative_success"] = float(rs)
    row["ci_rs_low"] = float(np.exp(ln_rs - Z * se_ln_rs))
    row["ci_rs_high"] = float(np.exp(ln_rs + Z * se_ln_rs))
    row["rs_p_value"] = float(chi2.sf((ln_rs / se_ln_rs) ** 2, df=1))
    # A Wald test on the log odds ratio, the P value that goes with the Woolf interval above.
    # `p_value` is Fisher's exact; the two disagree in the tails, and Supplementary Results 11.4
    # reports this one for the Pharmaprojects genetic-support annotation.
    row["z_p_value"] = float(2 * norm.sf(abs(ln_or / se_ln_or)))
    return row


def support_mask(
    df: pd.DataFrame,
    pav: bool = False,
    ta_min: int | None = None,
    ta_max: int | None = None,
    gps_min: int | None = None,
    gps_max: int | None = None,
    score_column: str | None = None,
) -> pd.Series:
    """Genetic-support mask for one (PAV, pleiotropy window) definition.

    Bounds are inclusive; None means unbounded. Any pleiotropy bound implies the target must be
    present in `genes_therapeutic_areas` (`in_gps`), which is what the published inner join did.

    Args:
        df: pair-level master table
        pav: require a protein-altering variant in the supporting credible set
        ta_min: minimum number of therapeutic areas for the target
        ta_max: maximum number of therapeutic areas for the target
        gps_min: minimum number of distinct diseases (gPS) for the target
        gps_max: maximum number of distinct diseases (gPS) for the target

    Returns:
        boolean Series aligned to df
    """
    if score_column is None:
        score_column = "score_pav" if pav else "score_all"
    mask = df[score_column].notna()
    bounds = [
        ("uniqueTherapeuticAreas", ta_min, ta_max),
        ("uniqueDiseases", gps_min, gps_max),
    ]
    if any(lo is not None or hi is not None for _, lo, hi in bounds):
        mask = mask & df["in_gps"]
    for column, lo, hi in bounds:
        if lo is not None:
            mask = mask & (df[column] >= lo)
        if hi is not None:
            mask = mask & (df[column] <= hi)
    return mask


def window_label(ta_min: int | None, ta_max: int | None) -> str:
    """Human-readable label for a therapeutic-area window."""
    if ta_min is None and ta_max is None:
        return "all"
    if ta_max is None:
        return f">={ta_min}"
    if ta_min is None:
        return f"<={ta_max}"
    if ta_min == ta_max:
        return f"{ta_min}"
    return f"{ta_min}-{ta_max}"


def contrast(x_low: int, n_low: int, x_high: int, n_high: int) -> dict:
    """Wald test of two supported strata against each other on the log-odds scale.

    This is the significance bracket in Figure 5b. A single logistic model with "no genetic
    support" as the reference gives both strata the same reference coefficient, so it cancels
    from their difference and the contrast reduces to the log odds ratio of the two 2x2 rows.

    Args:
        x_low: approved pairs in the low stratum
        n_low: non-approved pairs in the low stratum
        x_high: approved pairs in the high stratum
        n_high: non-approved pairs in the high stratum

    Returns:
        dict with the log odds ratio, its standard error, z and the two-sided P
    """
    odds_low = x_low / n_low
    odds_high = x_high / n_high
    log_or = np.log(odds_low / odds_high)
    se = np.sqrt(1 / x_low + 1 / n_low + 1 / x_high + 1 / n_high)
    z = log_or / se
    return {
        "log_or": float(log_or),
        "se": float(se),
        "z": float(z),
        "p_value": float(2 * norm.sf(abs(z))),
        "odds_low": float(odds_low),
        "odds_high": float(odds_high),
    }


def bh_plain(pvalues) -> np.ndarray:
    """Step-up Benjamini-Hochberg multiplier p * m / rank, without monotone enforcement.

    This is the convention the published Figure 5b FDRs follow, which is why the rare-variant
    FDR slightly exceeds the gPS FDR although their raw P values are almost identical.
    """
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(p) + 1)
    return np.minimum(p * len(p) / rank, 1.0)


def pleiotropy_success_curves(
    frame: pd.DataFrame,
    pleiotropy: str,
    n_grid: int = 200,
    n_boot: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Probability-of-success curves against a pleiotropy measure, for Figure 5c.

    Two solid lines come from one logistic fit on the whole table, quadratic in log
    pleiotropy, evaluated with and without genetic support. The dashed observed line and all
    three ribbons come from `n_boot` bootstrap resamples: the model ribbons are percentile
    intervals of the refitted predictions, and the observed line is the mean of a lowess smooth
    of the supported pairs.

    This ran inside `figure_5.R` until the fits were moved upstream. The R and Python lowess
    agree exactly on identical resamples, but the two RNG streams do not, so the bootstrap
    quantities carry Monte Carlo error of roughly +/-0.02 in probability at `n_boot = 200`.
    The solid model curves are order- and implementation-independent.

    Args:
        frame: one row per target-indication pair, with `outcome`, `geneticSupport` and the
            pleiotropy column. Row order affects the bootstrap, so it must be sorted upstream.
        pleiotropy: name of the pleiotropy column
        n_grid: points on the log-spaced x grid
        n_boot: bootstrap resamples
        seed: seed for the resampling

    Returns:
        one row per grid point, with the two model curves, the observed curve and the
        2.5/97.5 percentile bounds of each
    """
    import statsmodels.api as sm
    from statsmodels.nonparametric.smoothers_lowess import lowess

    x_max = float(frame.loc[frame[pleiotropy] >= 1, pleiotropy].max())
    x_grid = np.exp(np.linspace(np.log(1.0), np.log(x_max), n_grid))

    def design(x, genetic_support):
        """Model matrix: intercept, genetic support, log pleiotropy and its square."""
        log_x = np.log(np.asarray(x, dtype=float) + 1.0)
        support = np.asarray(genetic_support, dtype=float) * np.ones(len(log_x))
        return sm.add_constant(np.column_stack([support, log_x, log_x**2]), has_constant="add")

    def fit(rows):
        matrix = design(rows[pleiotropy].to_numpy(), rows["geneticSupport"].to_numpy())
        return sm.GLM(rows["outcome"].to_numpy(), matrix, family=sm.families.Binomial()).fit()

    base = fit(frame)
    rng = np.random.default_rng(seed)
    n_rows = len(frame)
    boot_gs1 = np.full((n_boot, n_grid), np.nan)
    boot_gs0 = np.full((n_boot, n_grid), np.nan)
    boot_observed = np.full((n_boot, n_grid), np.nan)

    for i in range(n_boot):
        resample = frame.iloc[rng.integers(0, n_rows, n_rows)]
        try:
            model = fit(resample)
            boot_gs1[i] = model.predict(design(x_grid, 1))
            boot_gs0[i] = model.predict(design(x_grid, 0))
        except Exception:  # a resample can be separable; the published code skips it too
            pass
        supported = resample[(resample["geneticSupport"] == 1) & (resample[pleiotropy] >= 1)]
        if supported[pleiotropy].nunique() > 3:
            smooth = lowess(
                supported["outcome"].to_numpy(),
                supported[pleiotropy].to_numpy(),
                frac=0.3,
                it=3,
                return_sorted=True,
            )
            boot_observed[i] = np.interp(x_grid, smooth[:, 0], smooth[:, 1])

    def bounds(matrix):
        return np.nanquantile(matrix, 0.025, axis=0), np.nanquantile(matrix, 0.975, axis=0)

    gs1_lo, gs1_hi = bounds(boot_gs1)
    gs0_lo, gs0_hi = bounds(boot_gs0)
    obs_lo, obs_hi = bounds(boot_observed)

    return pd.DataFrame(
        {
            "pleiotropy": pleiotropy,
            "x": x_grid,
            "model_gs1": base.predict(design(x_grid, 1)),
            "model_gs0": base.predict(design(x_grid, 0)),
            "observed_gs1": np.nanmean(boot_observed, axis=0),
            "model_gs1_lo": gs1_lo,
            "model_gs1_hi": gs1_hi,
            "model_gs0_lo": gs0_lo,
            "model_gs0_hi": gs0_hi,
            "observed_lo": obs_lo,
            "observed_hi": obs_hi,
        }
    )

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

"""Clinical phase transition rates stratified by pleiotropy.

Used by Supplementary Results 10 and Extended Data Figure 8. A transition rate is the share of
target-indication pairs reaching the end phase among those reaching the start phase, with Wilson
intervals; groups are compared within each transition by two-sided proportions z-tests, adjusted
across all nine tests by Benjamini-Hochberg.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from manuscript_methods.enrichment import bh_plain

TRANSITIONS = [(1, 2, "Phase I→II"), (2, 3, "Phase II→III"), (3, 4, "Phase III→approval")]
COMPARISONS = [("med_vs_low", 1, 0), ("high_vs_low", 2, 0), ("high_vs_med", 2, 1)]
COMPARISON_LABELS = {
    "med_vs_low": "Medium vs Low",
    "high_vs_low": "High vs Low",
    "high_vs_med": "High vs Medium",
}

# The two pleiotropy scales the manuscript reports, with the bin edges each uses.
GROUPINGS = {
    "TAs": {
        "column": "uniqueTherapeuticAreas",
        "edges": [0, 1, 5, float("inf")],
        "labels": ["1 TA", "2–5 TAs", "≥6 TAs"],
    },
    "gPS": {
        "column": "uniqueDiseases",
        "edges": [0, 1, 9, float("inf")],
        "labels": ["gPS 1", "gPS 2–9", "gPS ≥10"],
    },
}


def universe(pairs: pd.DataFrame) -> pd.DataFrame:
    """The 18,480 target-indication pairs with at least one therapeutic area of genetic support."""
    return pairs[pairs["uniqueTherapeuticAreas"] >= 1].copy()


def assign_bins(data: pd.DataFrame, spec: dict) -> pd.Series:
    """Cut a pleiotropy column into Low, Medium and High bins."""
    return pd.cut(data[spec["column"]], spec["edges"], labels=spec["labels"])


def transition_table(data: pd.DataFrame, spec: dict, grouping: str) -> pd.DataFrame:
    """Per group per transition: pairs at the start phase, pairs reaching the end, rate and CI."""
    binned = assign_bins(data, spec)
    rows = []
    for tier, label in zip(["Low", "Medium", "High"], spec["labels"], strict=True):
        subset = data[binned == label]
        for start_phase, end_phase, name in TRANSITIONS:
            n_start = int((subset["maxClinicalPhase"] >= start_phase).sum())
            n_reach = int((subset["maxClinicalPhase"] >= end_phase).sum())
            ci_low, ci_high = proportion_confint(n_reach, n_start, alpha=0.05, method="wilson")
            rows.append(
                {
                    "grouping": grouping,
                    "tier": tier,
                    "group": label,
                    "group_n": len(subset),
                    "transition": name,
                    "n_at_start": n_start,
                    "n_reaching": n_reach,
                    "rate": n_reach / n_start if n_start else np.nan,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def pairwise_table(rates: pd.DataFrame, spec: dict, grouping: str) -> pd.DataFrame:
    """Chi-square omnibus per transition, pairwise z-tests, risk ratios and BH-adjusted P values."""
    indexed = rates.set_index(["group", "transition"])
    rows = []
    for _, _, name in TRANSITIONS:
        counts = [
            [
                int(indexed.loc[(label, name), "n_reaching"]),
                int(indexed.loc[(label, name), "n_at_start"] - indexed.loc[(label, name), "n_reaching"]),
            ]
            for label in spec["labels"]
        ]
        _, omnibus, _, _ = chi2_contingency(counts)
        for key, a, b in COMPARISONS:
            reach_a, start_a = counts[a][0], counts[a][0] + counts[a][1]
            reach_b, start_b = counts[b][0], counts[b][0] + counts[b][1]
            _, p_raw = proportions_ztest([reach_a, reach_b], [start_a, start_b])
            rate_a, rate_b = reach_a / start_a, reach_b / start_b
            risk_ratio = rate_a / rate_b
            se = np.sqrt((1 - rate_a) / (start_a * rate_a) + (1 - rate_b) / (start_b * rate_b))
            rows.append(
                {
                    "grouping": grouping,
                    "transition": name,
                    "comparison_key": key,
                    "comparison": COMPARISON_LABELS[key],
                    "omnibus_p": omnibus,
                    "risk_ratio": risk_ratio,
                    "rr_ci_low": risk_ratio * np.exp(-1.96 * se),
                    "rr_ci_high": risk_ratio * np.exp(1.96 * se),
                    "p_raw": p_raw,
                }
            )
    table = pd.DataFrame(rows)
    # The published Extended Data Figure 8 used statsmodels' monotone BH; Figure 5b uses the plain
    # multiplier. Both are carried because the two conventions disagree on other families.
    table["p_adj_bh"] = multipletests(table["p_raw"], method="fdr_bh")[1]
    table["p_adj_bh_plain"] = bh_plain(table["p_raw"])
    return table


def both_scales(pairs: pd.DataFrame) -> tuple:
    """Rate and test tables for both pleiotropy scales, over the shared universe."""
    data = universe(pairs)
    rates = {name: transition_table(data, spec, name) for name, spec in GROUPINGS.items()}
    tests = {name: pairwise_table(rates[name], GROUPINGS[name], name) for name in GROUPINGS}
    return rates, tests

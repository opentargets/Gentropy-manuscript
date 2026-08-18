"""Fig. 5b therapeutic-area pleiotropy stratum: enrichment rows, contrast, FDR family.

Referee comment R2-MJ-1 asks for the number-of-TAs pleiotropy axis to be shown next to gPS in
Fig. 5b. This script produces everything that panel and the surrounding text need:

1. the two new forest rows (`any` genetic support, PAV not required, TA == 1 and TA >= 6) from the
   pair-level master table, matching `or10_phase0_grid_full-r1.csv`;
2. the within-group contrast between those two strata, computed the way the published panel does it
   -- one logistic model on all 37,377 pairs with "no genetic support" as the common reference, then
   a Wald test of the two supported strata against each other. The reference level cancels, so the
   closed-form log-odds-ratio contrast is reported alongside the fitted model as a cross-check;
3. the same contrast for the published gPS pair, to confirm the method reproduces P = 0.008;
4. Benjamini-Hochberg FDR over the within-group tests of Fig. 5b, before and after the TA test
   joins the family (m = 4 -> m = 5).

Inputs
    data/intermediate_files/ti_pairs_chembl_master-r1.parquet   (01_build_pair_tables.ipynb)
    data/intermediate_files/or10_phase0_grid_full-r1.csv        (02_phase0_threshold_grid.ipynb)

Outputs
    data/intermediate_files/fig5b_ta_rows-r1.csv        forest rows for the new TA group
    data/intermediate_files/fig5b_ta_contrast-r1.csv    all within-group contrasts
    data/intermediate_files/fig5b_fdr_family-r1.csv     BH FDR before/after adding the TA test
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "intermediate_files"
sys.path.insert(0, str(REPO / "chapters" / "06-review-r1" / "or10-optimism-validation"))

from or10_stats import or_rs, support_mask  # noqa: E402

Z = 1.96

# Fig. 5b pleiotropy strata. `low`/`high` are the two rows that carry the significance bracket.
STRATA = {
    "TA": {
        "low": {"label": "TAs=1", "kwargs": {"ta_min": 1, "ta_max": 1}},
        "high": {"label": "TAs>=6", "kwargs": {"ta_min": 6, "ta_max": None}},
    },
    "gPS": {
        "low": {"label": "gPS<=5", "kwargs": {"gps_min": 1, "gps_max": 5}},
        "high": {"label": "gPS>=10", "kwargs": {"gps_min": 10, "gps_max": None}},
    },
}


def closed_form_contrast(x_low: int, n_low: int, x_high: int, n_high: int) -> dict:
    """Wald test of two supported strata against each other on the log-odds scale.

    With a single logistic model whose reference is "no genetic support", the reference
    coefficient enters both stratum log-odds identically and cancels from the difference, so the
    contrast reduces to the log odds ratio of the two 2x2 supported rows.

    Args:
        x_low: approved pairs in the low-pleiotropy stratum
        n_low: non-approved pairs in the low-pleiotropy stratum
        x_high: approved pairs in the high-pleiotropy stratum
        n_high: non-approved pairs in the high-pleiotropy stratum

    Returns:
        dict with the log odds ratio, its standard error, z and two-sided P
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


def fitted_contrast(master: pd.DataFrame, low_mask: pd.Series, high_mask: pd.Series) -> dict:
    """Same contrast from the three-level logistic model the published panel fits.

    Pairs are coded E = None (no genetic support), Low, High; pairs supported but in neither
    pleiotropy stratum are dropped so the two coefficients are the strata of interest, exactly as
    `drug_enrichemnt_from_evidence_log_regression` codes `GeneralOnly` versus `subEvidence`.

    Args:
        master: pair-level master table
        low_mask: boolean Series, support in the low-pleiotropy stratum
        high_mask: boolean Series, support in the high-pleiotropy stratum

    Returns:
        dict with the fitted contrast estimate, standard error, z and two-sided P
    """
    supported_any = support_mask(master)
    e = np.where(low_mask, "Low", np.where(high_mask, "High", "None"))
    keep = ~(supported_any & ~(low_mask | high_mask))
    df = pd.DataFrame({"outcome": master["approved"].to_numpy(), "E": e})[keep.to_numpy()]
    df["E"] = pd.Categorical(df["E"], categories=["None", "Low", "High"], ordered=False)

    fit = smf.logit("outcome ~ C(E)", data=df).fit(disp=0)
    # coefficients are [Intercept, C(E)[T.Low], C(E)[T.High]]; test Low - High
    ct = fit.t_test(np.array([0.0, 1.0, -1.0]))
    return {
        "fit_log_or": float(np.ravel(ct.effect)[0]),
        "fit_se": float(np.ravel(ct.sd)[0]),
        "fit_z": float(np.ravel(ct.statistic)[0]),
        "fit_p_value": float(np.ravel(ct.pvalue)[0]),
        "n_model_rows": int(len(df)),
    }


def forest_row(master: pd.DataFrame, label: str, kwargs: dict) -> dict:
    """One Fig. 5b forest row: OR, CI, relative success and the 2x2 counts."""
    row = or_rs(support_mask(master, pav=False, **kwargs), master["approved"])
    row["datasource"] = label
    return row


def main() -> None:
    """Compute the TA rows, the within-group contrasts and the BH FDR family."""
    master = pd.read_parquet(DATA / "ti_pairs_chembl_master-r1.parquet")
    print(f"pairs: {len(master)}  approved: {int(master['approved'].sum())}")
    assert len(master) == 37377, len(master)

    # ---- sanity check against the published widest window ----
    widest = or_rs(support_mask(master, pav=False, ta_min=1), master["approved"])
    print(
        f"any + TA>=1: supported {widest['n_support']} pairs, "
        f"{widest['yes_evid-high_clinphase']} approved"
    )
    assert widest["n_support"] == 742, widest["n_support"]
    assert widest["yes_evid-high_clinphase"] == 242, widest["yes_evid-high_clinphase"]

    # ---- new forest rows ----
    ta_rows = pd.DataFrame(
        [forest_row(master, s["label"], s["kwargs"]) for s in STRATA["TA"].values()]
    )
    grid = pd.read_csv(DATA / "or10_phase0_grid_full-r1.csv")
    grid = grid[(~grid["pav"]) & (grid["clinicalPhase"] == "4+")]
    for (_, row), window in zip(ta_rows.iterrows(), ["1", ">=6"], strict=True):
        ref = grid[grid["window"] == window].iloc[0]
        assert np.isclose(row["odds_ratio"], ref["odds_ratio"]), (row["datasource"], window)
        assert row["yes_evid-high_clinphase"] == ref["yes_evid-high_clinphase"]
    print("TA rows agree with or10_phase0_grid_full-r1.csv")
    print(
        ta_rows[
            [
                "datasource",
                "odds_ratio",
                "ci_low",
                "ci_high",
                "relative_success",
                "yes_evid-low_clinphase",
                "yes_evid-high_clinphase",
            ]
        ].to_string(index=False)
    )

    # ---- within-group contrasts ----
    contrasts = []
    for metric, spec in STRATA.items():
        low_mask = support_mask(master, pav=False, **spec["low"]["kwargs"])
        high_mask = support_mask(master, pav=False, **spec["high"]["kwargs"])
        approved = master["approved"].to_numpy().astype(bool)
        counts = {
            "x_low": int((low_mask & approved).sum()),
            "n_low": int((low_mask & ~approved).sum()),
            "x_high": int((high_mask & approved).sum()),
            "n_high": int((high_mask & ~approved).sum()),
        }
        contrasts.append(
            {
                "metric": metric,
                "low": spec["low"]["label"],
                "high": spec["high"]["label"],
                **counts,
                **closed_form_contrast(**counts),
                **fitted_contrast(master, low_mask, high_mask),
            }
        )
    contrast_df = pd.DataFrame(contrasts)
    print()
    print(
        contrast_df[
            ["metric", "x_low", "n_low", "x_high", "n_high", "log_or", "se", "z", "p_value",
             "fit_log_or", "fit_se", "fit_p_value"]
        ].to_string(index=False)
    )

    ta_p = float(contrast_df.loc[contrast_df["metric"] == "TA", "fit_p_value"].iloc[0])
    gps_p = float(contrast_df.loc[contrast_df["metric"] == "gPS", "fit_p_value"].iloc[0])
    print(f"\nTA 1 vs >=6 contrast:  P = {ta_p:.4f}")
    print(f"gPS <=5 vs >=10 contrast: P = {gps_p:.4f}  (published P = 0.008)")

    # ---- BH FDR over the Fig. 5b within-group tests ----
    # Published raw P values, from drug_enrichment_subsets_vs_full_l2g.csv (diffence_pval) for PAV,
    # rare and effect size, and from the gPS contrast above. The published FDRs (0.001, 0.015, 0.01)
    # are plain step-up BH, p * m / rank, without the usual monotone enforcement -- which is why the
    # published rare FDR (0.015) exceeds the published gPS FDR (0.01) even though the raw P values
    # are almost identical. `bh_plain` keeps that convention so the printed manuscript values stay
    # reproducible; `bh_monotone` is the textbook enforced-monotone version, reported for reference.
    published = [
        ("PAV vs non-PAV", 0.000244095288211),
        ("rare vs common", 0.00771735759091),
        ("gPS <=5 vs >=10", gps_p),
        ("effect size", 0.193313469827),
    ]
    family = published + [("TA 1 vs >=6", ta_p)]

    def bh_plain(pvals: list[float]) -> np.ndarray:
        """Step-up BH multiplier p * m / rank, no monotone enforcement (published convention)."""
        p = np.asarray(pvals, dtype=float)
        order = np.argsort(p, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(1, len(p) + 1)
        return np.minimum(p * len(p) / rank, 1.0)

    fdr_df = pd.DataFrame(
        {
            "comparison": [name for name, _ in family],
            "p_raw": [p for _, p in family],
            "bh_plain_m4": list(bh_plain([p for _, p in published])) + [np.nan],
            "bh_plain_m5": bh_plain([p for _, p in family]),
            "bh_monotone_m4": list(
                multipletests([p for _, p in published], method="fdr_bh")[1]
            )
            + [np.nan],
            "bh_monotone_m5": multipletests([p for _, p in family], method="fdr_bh")[1],
        }
    )
    print()
    print(fdr_df.to_string(index=False))
    print(
        "\nrare and gPS raw P differ only at the 6th significant digit "
        f"({published[1][1]:.9f} vs {gps_p:.9f}); rare keeps rank 2 and gPS rank 3, "
        "as in the published family"
    )

    ta_rows.to_csv(DATA / "fig5b_ta_rows-r1.csv", index=False)
    contrast_df.to_csv(DATA / "fig5b_ta_contrast-r1.csv", index=False)
    fdr_df.to_csv(DATA / "fig5b_fdr_family-r1.csv", index=False)

    # ---- augmented forest-plot input for figure_5.R ----
    # The published table is left untouched; figure_5.R prefers this `-r1` copy when it exists.
    base = pd.read_csv(DATA / "drug_enrichment_subsets_vs_full_l2g.csv")
    new = ta_rows.rename(columns={"relative_success": "Relative success"}).assign(
        datasource=["TA-1_subEvid", "TA-6plus_subEvid"],
        drugsource="full_chembl",
        total_indirect_assoc=base["total_indirect_assoc"].iloc[0],
        diffence_pval=ta_p,
    )
    augmented = pd.concat([base, new[base.columns]], ignore_index=True)
    augmented.to_csv(DATA / "drug_enrichment_subsets_vs_full_l2g-r1.csv", index=False)
    print(
        f"\nwrote fig5b_ta_rows-r1.csv, fig5b_ta_contrast-r1.csv, fig5b_fdr_family-r1.csv and "
        f"drug_enrichment_subsets_vs_full_l2g-r1.csv ({len(augmented)} rows) to {DATA}"
    )


if __name__ == "__main__":
    main()

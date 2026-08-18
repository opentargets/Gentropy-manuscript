"""Figure 3a/b data on the resolved trait column.

Ports `chapters/03-manuscript-figures/figure_3/python_scripts/prepare_plot_a_b_data.py` off Spark
and onto `diseaseIds`. Every statistical choice is kept identical: same representative-variant
selection, same min-max covariate scaling, same `ncx2.sf(x=32.84125, df=1, nc=ncp)` predicted power,
same negative-binomial fits (`disp=False, maxiter=1000`), same MAF bins.

Note on the two output names: the upstream script deliberately swaps them --
`data/plot_a.csv` holds the MAF-bin means (Figure 3a) and `data/plot_b.csv` holds the regression
coefficients (Figure 3b). That convention is preserved here.

CONTROL. Asserted against the committed `data/plot_a.csv` and `data/plot_b.csv` -- the artefacts
`figure_3.R` actually reads. Despite the upstream script pointing at a GCS copy of
`variant_pleiotropy`, the local `figure_3/python_scripts/variant_pleiotropy` turns out to be that
same vintage: the raw column reproduces all six univariate coefficients and all seven MAF bins to
~1e-13.

Two independent vintages of this regression exist in the repo, and the committed figure data comes
from the older one. `figure_3/python_scripts/clustering_analysis.ipynb` printed R2 = 0.17732233,
while `02-analysis/04-variant-level-ps/02_clustering_analysis.ipynb` printed 0.17975229. The
manuscript quotes 17.7% and 6.0%, matching the figure vintage; this script reproduces that vintage.
The 02-analysis notebook's printed coefficients (maxAbsBeta 2.151716, predictedPower 1.444744) are
therefore stale relative to the published figure and are reported for reference only.

Run from the repository root.
"""

import collections
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import statsmodels.api as sm
from scipy.stats import ncx2

sys.path.insert(0, "chapters/06-review-r1/cluster-disease-vs-ta")
import cluster_lib_r1 as lib  # noqa: E402

VARIANT_TABLE = "chapters/03-manuscript-figures/figure_3/python_scripts/variant_pleiotropy"
FIGURE_DATA = "chapters/03-manuscript-figures/figure_3/data/"

COVARIATES = [
    "maxAbsBetaNormalised",
    "maxMAFNormalised",
    "maxEffectiveSampleSizeNormalised",
    "gerpNormalisedNormalised",
    "vepBinaryNormalised",
    "predictedPowerNormalised",
]
LABELS = {
    "maxAbsBetaNormalised": "Absolute beta",
    "maxMAFNormalised": "MAF",
    "maxEffectiveSampleSizeNormalised": "Sample size",
    "gerpNormalisedNormalised": "GERP",
    "vepBinaryNormalised": "PAV",
    "predictedPowerNormalised": "Predicted power",
}
BINS = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
BIN_LABELS = ["0-0.01", "0.01-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5"]

# --- clusters and representative variants -------------------------------------
cs = lib.load_credible_sets()
edges = lib.load_edges(set(cs.studyLocusId))
clusters = lib.cluster(list(zip(cs.studyLocusId, cs.variantId)), edges)
print(f"clusters: {len(clusters)}")

locus_variant = dict(zip(cs.studyLocusId, cs.variantId))
trait_columns = {"raw": dict(zip(cs.studyLocusId, cs.raw)), "resolved": dict(zip(cs.studyLocusId, cs.resolved))}

variants = ds.dataset(VARIANT_TABLE, format="parquet").to_table(
    columns=["variantId", "maxAbsBeta", "maxMAF", "maxEffectiveSampleSize", "maxVarG",
             "gerpNormalised", "vepScore"]
).to_pandas().drop_duplicates("variantId")


def build_cluster_frame(column):
    """Reproduce `cluster_pleiotropy` for one trait column."""
    trait_lookup = trait_columns[column]
    rows = []
    for _, members in clusters:
        per_variant = collections.defaultdict(set)
        for locus_id in members:
            per_variant.setdefault(locus_variant[locus_id], set())
            traits = trait_lookup.get(locus_id)
            if traits is not None:
                per_variant[locus_variant[locus_id]].update(traits)
        # row_number() over (desc uniqueTraitCountForVariant, asc leadVariantId) == 1
        representative = sorted(per_variant.items(), key=lambda kv: (-len(kv[1]), kv[0]))[0][0]
        all_traits = set()
        for trait_set in per_variant.values():
            all_traits |= trait_set
        rows.append({
            "cluster_size": len(members),
            "uniqueTraitsInCluster": len(all_traits),
            "clusterVariantId": representative,
        })
    frame = pd.DataFrame(rows).merge(variants, left_on="clusterVariantId", right_on="variantId", how="inner")

    frame["ncp"] = (frame.maxAbsBeta ** 2 * frame.maxEffectiveSampleSize * frame.maxVarG) / 11
    frame["predictedPower"] = ncx2.sf(x=32.84125, df=1, nc=frame.ncp)

    frame["gerpNormalised"] = frame.gerpNormalised.fillna(frame.gerpNormalised.mean())
    frame["vepBinary"] = (frame.vepScore >= 0.66).astype(int)
    for col in ["maxAbsBeta", "maxMAF", "gerpNormalised", "vepBinary",
                "maxEffectiveSampleSize", "predictedPower"]:
        span = frame[col].max() - frame[col].min()
        frame[f"{col}Normalised"] = 0.0 if span == 0 else (frame[col] - frame[col].min()) / span
    return frame


def fit(frame):
    """Univariate and joint negative-binomial fits, plus the two MAF-bin prediction columns."""
    y = frame["uniqueTraitsInCluster"]
    records = []
    for covariate in COVARIATES:
        model = sm.NegativeBinomial(y, sm.add_constant(frame[[covariate]].copy())).fit(disp=False, maxiter=1000)
        ci = model.conf_int()
        records.append({"covariate": covariate, "model_type": "Univariate",
                        "coefficient": model.params[covariate], "std_error": model.bse[covariate],
                        "p_value": model.pvalues[covariate],
                        "ci_lower": ci.loc[covariate, 0], "ci_upper": ci.loc[covariate, 1]})

    x_multi = sm.add_constant(frame[COVARIATES].copy())
    joint = sm.NegativeBinomial(y, x_multi).fit(disp=False, maxiter=1000)
    ci = joint.conf_int()
    for covariate in COVARIATES:
        records.append({"covariate": covariate, "model_type": "Multi",
                        "coefficient": joint.params[covariate], "std_error": joint.bse[covariate],
                        "p_value": joint.pvalues[covariate],
                        "ci_lower": ci.loc[covariate, 0], "ci_upper": ci.loc[covariate, 1]})

    no_power = [c for c in COVARIATES if c != "predictedPowerNormalised"]
    joint_np = sm.NegativeBinomial(y, sm.add_constant(frame[no_power].copy())).fit(disp=False, maxiter=1000)

    coefficients = pd.DataFrame(records)
    coefficients["covariate_label"] = coefficients.covariate.map(LABELS)
    y_map = {c: i for i, c in enumerate(COVARIATES)}
    coefficients["y_numerical"] = coefficients.covariate.map(y_map)
    coefficients["y_plot"] = coefficients.y_numerical + np.where(
        coefficients.model_type == "Univariate", -0.1, 0.1)
    coefficients = coefficients[["covariate", "covariate_label", "model_type", "coefficient",
                                 "std_error", "p_value", "ci_lower", "ci_upper",
                                 "y_numerical", "y_plot"]]

    binned = frame.copy()
    binned["predicted_traits_full_model"] = joint.predict(x_multi)
    binned["predicted_traits_no_power"] = joint_np.predict(sm.add_constant(frame[no_power].copy()))
    binned["maxMAF_bin"] = pd.cut(binned.maxMAF, bins=BINS, labels=BIN_LABELS, right=False)
    bins = binned.groupby("maxMAF_bin", observed=False).agg(
        observed_mean=("uniqueTraitsInCluster", "mean"),
        observed_sem=("uniqueTraitsInCluster", "sem"),
        predicted_full_mean=("predicted_traits_full_model", "mean"),
        predicted_full_sem=("predicted_traits_full_model", "sem"),
        predicted_no_power_mean=("predicted_traits_no_power", "mean"),
        predicted_no_power_sem=("predicted_traits_no_power", "sem"),
    ).reset_index()
    bins["maxMAF_bin"] = bins.maxMAF_bin.astype(str)

    # The four R2 values the manuscript quotes (04_variant_pleiotropy.tex:24-26).
    r2 = {
        "full": float(np.corrcoef(y, joint.predict(x_multi))[0, 1] ** 2),
        "no_power": float(np.corrcoef(y, joint_np.predict(sm.add_constant(frame[no_power].copy())))[0, 1] ** 2),
    }
    for key, covariate in (("power_alone", "predictedPowerNormalised"),
                          ("sample_size_alone", "maxEffectiveSampleSizeNormalised")):
        x_one = sm.add_constant(frame[[covariate]].copy())
        model_one = sm.NegativeBinomial(y, x_one).fit(disp=False, maxiter=1000)
        r2[key] = float(np.corrcoef(y, model_one.predict(x_one))[0, 1] ** 2)
    return coefficients, bins, r2


results = {}
for column in ("raw", "resolved"):
    frame = build_cluster_frame(column)
    print(f"\n{column}: {len(frame)} clusters joined to the variant table, "
          f"vPS mean {frame.uniqueTraitsInCluster.mean():.6f}")
    results[column] = fit(frame)

# --- control: raw against the committed figure data ---------------------------
raw_coef, raw_bins, raw_r2 = results["raw"]
uni = raw_coef[raw_coef.model_type == "Univariate"].set_index("covariate")
committed_a = pd.read_csv(FIGURE_DATA + "plot_a.csv")
committed_b = pd.read_csv(FIGURE_DATA + "plot_b.csv")

print("\ncontrol: raw column vs the committed figure data (what figure_3.R reads)")
ok = True
for field in ("coefficient", "std_error", "p_value", "ci_lower", "ci_upper"):
    merged = committed_b.merge(raw_coef, on=["covariate", "model_type"], suffixes=("_c", "_m"))
    assert len(merged) == len(committed_b), "row set differs from plot_b.csv"
    worst = float((merged[f"{field}_c"] - merged[f"{field}_m"]).abs().max())
    good = worst < 1e-8
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  plot_b.csv {field:<12} max abs diff over 12 rows: {worst:.3e}")
for field in ("observed_mean", "observed_sem", "predicted_full_mean",
              "predicted_full_sem", "predicted_no_power_mean", "predicted_no_power_sem"):
    merged = committed_a.merge(raw_bins, on="maxMAF_bin", suffixes=("_c", "_m"))
    assert len(merged) == len(committed_a), "bin set differs from plot_a.csv"
    worst = float((merged[f"{field}_c"] - merged[f"{field}_m"]).abs().max())
    good = worst < 1e-8
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  plot_a.csv {field:<24} max abs diff over 7 bins: {worst:.3e}")

MANUSCRIPT_R2 = {"full": 0.177, "no_power": 0.060, "power_alone": 0.147, "sample_size_alone": 0.0045}
print("\n  manuscript R2 values (04_variant_pleiotropy.tex:24-26), raw column:")
for key, want in MANUSCRIPT_R2.items():
    got = raw_r2[key]
    good = abs(got - want) < 0.001
    ok &= good
    print(f"    {'PASS' if good else 'FAIL'}  {key:<18} {got:.4%}  vs published {want:.2%}")

print("\ncontrol PASSED" if ok else "\ncontrol FAILED")
if not ok:
    raise SystemExit("control failed -- not writing outputs")

# --- committed CSVs, for the record ------------------------------------------
cmp = (committed_b[committed_b.model_type == "Univariate"].set_index("covariate")["coefficient"]
       .rename("committed_plot_b").to_frame()
       .join(uni["coefficient"].rename("local_raw"))
       .join(results["resolved"][0].query("model_type == 'Univariate'")
             .set_index("covariate")["coefficient"].rename("local_resolved")))
cmp["raw_vs_committed"] = cmp.local_raw - cmp.committed_plot_b
cmp["resolved_vs_raw"] = cmp.local_resolved - cmp.local_raw
print("\nunivariate coefficients: committed plot_b.csv vs local raw vs local resolved")
print(cmp.to_string())

# --- write -------------------------------------------------------------------
res_coef, res_bins, res_r2 = results["resolved"]
res_bins.to_csv(FIGURE_DATA + "plot_a-r1.csv", index=False)
res_coef.to_csv(FIGURE_DATA + "plot_b-r1.csv", index=False)
cmp.to_csv(lib.INTERMEDIATE + "figure_3_column_diff-r1.csv")
print(f"\nwrote {FIGURE_DATA}plot_a-r1.csv  (Figure 3a, MAF bins)")
print(f"wrote {FIGURE_DATA}plot_b-r1.csv  (Figure 3b, coefficients)")
print(f"wrote {lib.INTERMEDIATE}figure_3_column_diff-r1.csv")

print("\nR2, raw -> resolved (published value in brackets):")
for key, published in MANUSCRIPT_R2.items():
    print(f"  {key:<18} {raw_r2[key]:.4%} -> {res_r2[key]:.4%}   (published {published:.2%})")
print("\nFigure 3a MAF bins, observed mean:")
print(raw_bins[["maxMAF_bin", "observed_mean"]].rename(columns={"observed_mean": "raw"})
      .assign(resolved=res_bins.observed_mean.values,
              delta=res_bins.observed_mean.values - raw_bins.observed_mean.values).to_string(index=False))

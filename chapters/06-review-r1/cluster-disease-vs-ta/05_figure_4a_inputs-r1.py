"""Stage Figure 4a's temporal vPS input on the resolved column.

Figure 4a reads `data/figure_4/Fig4A_stats_variant_pleiotropy.csv` -- the same table as
`cluster_stats_by_year.csv`, written tab-separated with a leading index column. This script emits the
resolved-column equivalent in that exact format.

CONTROL. The raw columns of `cluster_stats_by_year-r1.csv` must equal the committed
`Fig4A_stats_variant_pleiotropy.csv` field for field, on every year. That is asserted here against
the figure's own input file (not just against `cluster_stats_by_year.csv`), because it is the file
`figure_4.R` actually consumes.

Run from the repository root, after `03_temporal_vps-r1.py`.
"""

import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "chapters/06-review-r1/cluster-disease-vs-ta")
import cluster_lib_r1 as lib  # noqa: E402

FIGURE_DIR = "data/figure_4/"
COMMITTED = FIGURE_DIR + "Fig4A_stats_variant_pleiotropy.csv"
OUT = FIGURE_DIR + "Fig4A_stats_variant_pleiotropy-r1.csv"

committed = pd.read_csv(COMMITTED, sep="\t", index_col=0)
mine = pd.read_csv(lib.INTERMEDIATE + "cluster_stats_by_year-r1.csv")
print(f"committed rows: {len(committed)} ({committed.year.min()}-{committed.year.max()}), "
      f"regenerated rows: {len(mine)} ({mine.year.min()}-{mine.year.max()})")

# The committed figure input stops at 2024, matching the panel's coord_cartesian(xlim = c(2006, 2024)).
# Restrict to exactly the committed years so the resolved file is a drop-in replacement.
mine = mine[mine.year.isin(set(committed.year))].reset_index(drop=True)
print(f"restricted to the committed year range: {len(mine)} rows")

merged = committed.merge(mine, on="year", how="inner", suffixes=("_c", ""))
assert len(merged) == len(committed), "year sets differ"

print(f"\ncontrol: raw column vs {COMMITTED}")
ok = True
for field, mine_col in (("n_agg", "n_clusters"), ("n_count", "n_clusters"),
                        ("mean", "mean_raw"), ("sd", "sd_raw"), ("se", "se_raw")):
    if field.startswith("n_"):
        bad = merged[merged[field] != merged[mine_col]]
    else:
        bad = merged[~np.isclose(merged[field], merged[mine_col], rtol=0, atol=1e-9)]
    ok &= bad.empty
    print(f"  {'PASS' if bad.empty else f'FAIL ({len(bad)} years)'}  {field} vs {mine_col}")
    if not bad.empty:
        print(bad[["year", field, mine_col]].head(8).to_string(index=False))

if not ok:
    raise SystemExit("control failed -- not writing the resolved input")
print("\ncontrol PASSED: the regenerated raw column is the committed figure input.")

# Emit the resolved column in the committed file's exact shape and column order.
resolved = pd.DataFrame({
    "year": mine.year,
    "mean": mine.mean_resolved,
    "sd": mine.sd_resolved,
    "se": mine.se_resolved,
    "n_agg": mine.n_clusters,
    "n_count": mine.n_clusters,
})
assert list(resolved.columns) == list(committed.reset_index(drop=True).columns), \
    f"column order differs: {list(resolved.columns)} vs {list(committed.columns)}"
resolved.to_csv(OUT, sep="\t", index=True)
print(f"wrote {OUT}")

# The other three panel-a inputs are column-independent (gPS and coverage already use diseaseIds).
for name in ("Fig4A_stats_gene_pleiotropy.csv", "Fig4A_stats_gene_coverage.csv"):
    shutil.copy(FIGURE_DIR + name, FIGURE_DIR + name.replace(".csv", "-r1.csv"))
    print(f"copied unchanged: {name} -> {name.replace('.csv', '-r1.csv')}")

comparison = merged[["year", "mean"]].rename(columns={"mean": "raw"}).copy()
comparison["resolved"] = resolved["mean"].values
comparison["delta"] = comparison.resolved - comparison.raw
comparison["pct"] = comparison.delta / comparison.raw * 100
comparison.to_csv(lib.INTERMEDIATE + "figure_4a_column_diff-r1.csv", index=False)
print(f"\nwrote {lib.INTERMEDIATE}figure_4a_column_diff-r1.csv")
print(f"mean vPS shift: min {comparison.delta.min():+.6f}, max {comparison.delta.max():+.6f}, "
      f"worst {comparison.pct.abs().max():.3f}%")
print(comparison.to_string(index=False))

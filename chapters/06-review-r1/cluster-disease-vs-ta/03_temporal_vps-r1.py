"""Temporal vPS on the resolved trait column — Figure 4a variant-pleiotropy line.

Reproduces `chapters/02-analysis/05-gene-level-ps/02_temporal_vPS_gPS.ipynb` cells 6-20: join the
release study index's publicationDate, derive `year`, fill nulls with 2024, then for each year
2006-2025 re-cluster the credible sets published up to that year and take the mean, sd and se of the
per-cluster disease count.

Control: the raw column must reproduce the committed `cluster_stats_by_year.csv` row for row. Only
then is the resolved column reported. The gPS line in Figure 4a is untouched -- it already reads
`diseaseIds`.

Run from the repository root.
"""

import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "chapters/06-review-r1/cluster-disease-vs-ta")
import cluster_lib_r1 as lib  # noqa: E402

PUBLISHED = lib.INTERMEDIATE + "cluster_stats_by_year.csv"
OUT = lib.INTERMEDIATE + "cluster_stats_by_year-r1.csv"
DIFF = lib.INTERMEDIATE + "cluster_stats_by_year_column_diff-r1.csv"
YEARS = range(2006, 2026)

cs = lib.load_credible_sets(with_year=True)
print(f"credible sets: {len(cs)} (notebook: 70,618)")
print(f"year range: {cs.year.min()}-{cs.year.max()}, filled to 2024: {int((cs.year == 2024).sum())}")

# Edges are loaded once over all loci; each year's subset just filters both endpoints.
all_edges = lib.load_edges(set(cs.studyLocusId))
print(f"qualifying colocalisation edges: {len(all_edges)}")

records = []
for year in YEARS:
    subset = cs[cs.year <= year]
    if subset.empty:
        continue
    locus_ids = set(subset.studyLocusId)
    edges = [(l, r) for l, r in all_edges if l in locus_ids and r in locus_ids]
    clusters = lib.cluster(list(zip(subset.studyLocusId, subset.variantId)), edges)

    row = {"year": year, "n_credible_sets": len(subset), "n_clusters": len(clusters)}
    for label, column in (("raw", "raw"), ("resolved", "resolved")):
        diseases, _ = lib.cluster_counts(clusters, dict(zip(subset.studyLocusId, subset[column])))
        n = len(diseases)
        sd = float(np.std(diseases, ddof=1)) if n > 1 else 0.0
        row[f"mean_{label}"] = float(diseases.mean())
        row[f"sd_{label}"] = sd
        row[f"se_{label}"] = sd / math.sqrt(n) if n else None
    records.append(row)
    print(f"  {year}: {len(subset):>6} CSs -> {len(clusters):>6} clusters   "
          f"mean raw {row['mean_raw']:.4f}  resolved {row['mean_resolved']:.4f}")

stats = pd.DataFrame(records)

# --- control: raw must reproduce the committed CSV ----------------------------
published = pd.read_csv(PUBLISHED)
merged = published.merge(stats, on="year", how="inner", suffixes=("_pub", ""))
print(f"\ncontrol against {PUBLISHED} on {len(merged)} shared years")
failures = []
for field, mine in (("n_count", "n_clusters"), ("mean", "mean_raw"), ("sd", "sd_raw"), ("se", "se_raw")):
    if field == "n_count":
        bad = merged[merged[field] != merged[mine]]
    else:
        bad = merged[~np.isclose(merged[field], merged[mine], rtol=0, atol=1e-9)]
    status = "PASS" if bad.empty else f"FAIL ({len(bad)} years)"
    print(f"  {status}  {field} vs {mine}")
    if not bad.empty:
        failures.append((field, bad[["year", field, mine]]))

if failures:
    for field, bad in failures:
        print(f"\n  first mismatches for {field}:")
        print(bad.head(8).to_string(index=False))
    raise SystemExit("raw column does not reproduce the committed temporal table -- stopping")

print("\nraw column reproduces the committed table exactly.")

stats.to_csv(OUT, index=False)
diff = stats[["year", "n_clusters", "mean_raw", "mean_resolved"]].copy()
diff["delta"] = diff.mean_resolved - diff.mean_raw
diff["pct"] = diff.delta / diff.mean_raw * 100
diff.to_csv(DIFF, index=False)
print(f"wrote {OUT}\nwrote {DIFF}")
print(f"\nmean vPS shift, resolved - raw: min {diff.delta.min():+.4f}, max {diff.delta.max():+.4f}, "
      f"worst {diff.pct.abs().max():.3f}%")
print(diff.to_string(index=False))

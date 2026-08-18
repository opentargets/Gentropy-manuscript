"""Build Supplementary Table 15 — diseases linked through each colocalisation cluster.

Referee R2-MJ-12. `sections/extended_data.tex` carries `% TODO(data): populate tab:st15`, so this
table has never been built. Caption asks for: cluster identifier, lead variant(s), disease name,
EFO identifier, therapeutic area, and the cluster's variant pleiotropy score (vPS).

One row per (cluster, disease). Built on `diseaseIds`, the ontology-resolved trait column that the
gene-level pipeline has always used for gPS -- see README-r1.md for why the cluster-level code path
was the one outlier reading `traitFromSourceMappedIds`.

Control: rows per cluster must equal `uniqueTraitsInCluster_resolved` in
`data/intermediate_files/cluster_disease_ta_counts-r1.csv`, so this table cannot silently disagree
with supplementary figure SR6.

Run from the repository root.
"""

import sys

import pandas as pd

sys.path.insert(0, "chapters/06-review-r1/cluster-disease-vs-ta")
import cluster_lib_r1 as lib  # noqa: E402

OUT = lib.INTERMEDIATE + "st15_cluster_membership-r1.csv"

# --- clusters -----------------------------------------------------------------
cs = lib.load_credible_sets()
edges = lib.load_edges(set(cs.studyLocusId))
clusters = lib.cluster(list(zip(cs.studyLocusId, cs.variantId)), edges)
print(f"clusters: {len(clusters)}, credible sets: {sum(len(m) for _, m in clusters)}")

locus_variant = dict(zip(cs.studyLocusId, cs.variantId))
locus_traits = dict(zip(cs.studyLocusId, cs.resolved))

ta_of = lib.therapeutic_area_lookup()
name_of = lib.disease_names()

# --- one row per (cluster, disease) -------------------------------------------
rows = []
for cluster_id, (_, members) in enumerate(clusters):
    lead_variants = sorted({locus_variant[m] for m in members})
    diseases = set()
    for locus_id in members:
        trait_ids = locus_traits.get(locus_id)
        if trait_ids is not None:
            diseases.update(trait_ids)
    vps = len(diseases)
    lead_variant_field = ";".join(lead_variants)
    for disease_id in sorted(diseases):
        area = ta_of.get(disease_id)
        rows.append(
            {
                "cluster_id": cluster_id,
                "n_credible_sets": len(members),
                "n_lead_variants": len(lead_variants),
                "lead_variants": lead_variant_field,
                "diseaseName": name_of.get(disease_id),
                "diseaseId": disease_id,
                "therapeuticArea": lib.therapeutic_area_name(area) if area is not None else None,
                "therapeuticAreaId": area,
                "vPS": vps,
            }
        )

st15 = pd.DataFrame(rows)

# --- control against the SR6 table -------------------------------------------
counts = pd.read_csv(lib.INTERMEDIATE + "cluster_disease_ta_counts-r1.csv")
per_cluster = st15.groupby("cluster_id").size().rename("st15_rows")
check = counts.set_index("cluster_id")[["uniqueTraitsInCluster_resolved", "clusterNumberTherapeuticAreas_resolved"]].join(per_cluster)
mismatch = check[check["st15_rows"] != check["uniqueTraitsInCluster_resolved"]]
assert mismatch.empty, f"row count disagrees with SR6 table for {len(mismatch)} clusters"
print(f"PASS  rows per cluster == uniqueTraitsInCluster_resolved for all {len(check)} clusters")

ta_per_cluster = st15.groupby("cluster_id")["therapeuticAreaId"].nunique()
ta_mismatch = (ta_per_cluster != check["clusterNumberTherapeuticAreas_resolved"]).sum()
assert ta_mismatch == 0, f"TA count disagrees for {ta_mismatch} clusters"
print(f"PASS  distinct TAs per cluster == clusterNumberTherapeuticAreas_resolved for all {len(check)} clusters")

assert st15["diseaseId"].notna().all(), "null diseaseId"
assert st15["therapeuticAreaId"].notna().all(), "null therapeuticArea -- unresolvable id leaked in"
unnamed = int(st15["diseaseName"].isna().sum())
print(f"rows with no disease label in disease.parquet: {unnamed}")

st15.to_csv(OUT, index=False)
print(f"\nwrote {OUT}  {st15.shape}")
print(f"  clusters                 : {st15.cluster_id.nunique():,}")
print(f"  distinct diseases         : {st15.diseaseId.nunique():,}")
print(f"  distinct therapeutic areas: {st15.therapeuticAreaId.nunique()}")
print(f"  vPS range                 : {st15.vPS.min()}-{st15.vPS.max()}")
print("\nfirst rows:")
print(st15.head(8).to_string(index=False))
print("\nrows per therapeutic area:")
print(st15.therapeuticArea.value_counts().to_string())

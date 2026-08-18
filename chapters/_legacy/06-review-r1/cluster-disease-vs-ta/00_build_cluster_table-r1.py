"""Rebuild the cluster-level disease / TA count table for supplementary figure SR6.

Reproduces chapters/02-analysis/04-variant-level-ps/02_clustering_analysis.ipynb
cells 5-6 (clustering), 13 (uniqueTraitsInCluster) and 18-19 (clusterNumberTherapeuticAreas)
without Spark. Connected components are order-independent, so the cluster partition is
identical to the notebook's.

Emits BOTH trait-column variants per cluster:

  *_raw       from `traitFromSourceMappedIds`, the raw curator mapping the notebook reads.
              Reproduces the published 6,678 / range 1-122 / mean 2.16 exactly, so it is the
              control that proves this table has the same lineage as the manuscript's.
  *_resolved  from `diseaseIds`, the ontology-resolved mapping, which equals the study index's
              `diseaseIds` for all 70,618 CSs and is what every other analysis in the repo uses
              (variant-level uniqueDiseases, gene-level gPS, mappedTherapeuticAreas). The raw
              column carries 26 ids that no longer exist in disease.parquet (retired EFO ids,
              un-ingested Orphanet ids, two junk entries); Open Targets had already remapped all
              of them to live MONDO terms in `diseaseIds`. SR6 is drawn from the resolved counts.
"""

import pyarrow.dataset as ds
import pyarrow.compute as pc
import pandas as pd
import numpy as np

REL = "data/25.06/output/"
INT = "data/intermediate_files/"

# --- lead variants (notebook cell 5) -----------------------------------------
cs = ds.dataset(INT + "qualifying_credible_sets", format="parquet").to_table(
    columns={
        "studyLocusId": ds.field("studyLocusId"),
        "variantId": ds.field("variantId"),
        "traitFromSourceMappedIds": ds.field("traitFromSourceMappedIds"),
        "diseaseIds": ds.field("diseaseIds"),
        "pValueMantissa": pc.struct_field(ds.field("variantStatistics"), "pValueMantissa"),
        "pValueExponent": pc.struct_field(ds.field("variantStatistics"), "pValueExponent"),
        "leadVariantPIP": pc.struct_field(ds.field("locusStatistics"), "leadVariantPIP"),
    }
).to_pandas()
print("qualifying credible sets:", len(cs))

cs["pValue"] = cs["pValueMantissa"].astype(float) * np.power(10.0, cs["pValueExponent"].astype(float))
cs = cs.sort_values(["pValue", "leadVariantPIP"], ascending=[True, False], kind="mergesort")

study_loci = list(
    zip(cs["studyLocusId"], cs["variantId"], cs["traitFromSourceMappedIds"], cs["diseaseIds"])
)
locus_ids = set(cs["studyLocusId"])

# --- colocalisation edges (notebook cell 5) ----------------------------------
edges = []
for path, col, thr in (
    ("colocalisation_coloc", "h4", 0.8),
    ("colocalisation_ecaviar", "clpp", 0.01),
):
    d = ds.dataset(REL + path, format="parquet")
    n = 0
    for batch in d.to_batches(
        columns=["leftStudyLocusId", "rightStudyLocusId", col],
        filter=ds.field(col) >= thr,
    ):
        left = batch.column("leftStudyLocusId").to_pylist()
        right = batch.column("rightStudyLocusId").to_pylist()
        for l, r in zip(left, right):
            if l in locus_ids and r in locus_ids:
                edges.append((l, r))
                n += 1
    print(f"{path}: {n} qualifying edges")
print("total edges:", len(edges))

# --- clustering (notebook cell 6) --------------------------------------------
study_locus_info = {sl: (v, raw, res) for sl, v, raw, res in study_loci}

variant_to_loci = {}
for sl, v, _, _ in study_loci:
    variant_to_loci.setdefault(v, []).append(sl)

coloc_lookup = {}
for l, r in edges:
    coloc_lookup.setdefault(l, set()).add(r)
    coloc_lookup.setdefault(r, set()).add(l)

clusters = []
used = set()
for seed, _, _, _ in study_loci:
    if seed in used:
        continue
    queue = {seed}
    members = set()
    while queue:
        lid = queue.pop()
        if lid in used:
            continue
        members.add(lid)
        used.add(lid)
        for nb in coloc_lookup.get(lid, ()):
            if nb not in used:
                queue.add(nb)
        vid = study_locus_info[lid][0]
        for shared in variant_to_loci.get(vid, ()):
            if shared not in used:
                queue.add(shared)
    clusters.append((seed, sorted(members)))
print("clusters:", len(clusters))
print("total cluster size:", sum(len(m) for _, m in clusters))

# --- therapeutic-area lookup (notebook cell 18) ------------------------------
HIERARCHY = [
    "EFO_0001444", "MONDO_0045024", "EFO_0005741", "OTAR_0000009", "OTAR_0000014",
    "MONDO_0024458", "EFO_0000319", "EFO_0009605", "EFO_0000540", "EFO_0010282",
    "OTAR_0000017", "EFO_0010285", "EFO_0001379", "OTAR_0000010", "EFO_0009690",
    "OTAR_0000006", "MONDO_0021205", "EFO_0005803", "EFO_0000618", "MONDO_0002025",
    "OTAR_0000020", "OTAR_0000018", "EFO_0003765",
]

study_disease_ids = set()
for chunk in ds.dataset(REL + "study", format="parquet").to_batches(columns=["diseaseIds"]):
    for lst in chunk.column("diseaseIds").to_pylist():
        if lst:
            study_disease_ids.update(lst)
print("distinct study diseaseIds:", len(study_disease_ids))

disease = ds.dataset(REL + "disease", format="parquet").to_table(columns=["id", "ancestors"]).to_pydict()
efo_ta = {}
for did, anc in zip(disease["id"], disease["ancestors"]):
    if did not in study_disease_ids:
        continue                      # semi-join against study.diseaseIds
    anc = set(anc or ())
    ta = next((t for t in HIERARCHY if t in anc), None)
    efo_ta[did] = ta if ta is not None else "other"
print("efo -> TA lookup size:", len(efo_ta))

# --- per-cluster counts (notebook cells 13 and 19), both trait columns -------
# index 1 = traitFromSourceMappedIds (raw), index 2 = diseaseIds (resolved)
rows = []
for cid, (seed, members) in enumerate(clusters):
    counts = []
    for col_idx in (1, 2):
        traits = set()
        for lid in members:
            t = study_locus_info[lid][col_idx]
            if t is not None:
                traits.update(t)
        tas = {efo_ta[t] for t in traits if t in efo_ta}
        counts += [len(traits), len(tas)]
    rows.append((cid, seed, len(members), *counts))

df = pd.DataFrame(rows, columns=[
    "cluster_id", "lead_study_locus_id", "cluster_size",
    "uniqueTraitsInCluster_raw", "clusterNumberTherapeuticAreas_raw",
    "uniqueTraitsInCluster_resolved", "clusterNumberTherapeuticAreas_resolved",
])
df.to_csv(INT + "cluster_disease_ta_counts-r1.csv", index=False)
print("wrote", INT + "cluster_disease_ta_counts-r1.csv", df.shape)
print(df.drop(columns=["cluster_id", "lead_study_locus_id"]).describe().loc[["min", "max", "mean"]])

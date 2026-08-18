"""Shared clustering primitives for the review-round-1 cluster-column fix.

Faithful, Spark-free reimplementation of the colocalisation clustering in
`chapters/02-analysis/04-variant-level-ps/02_clustering_analysis.ipynb` (cells 5, 6, 18) and
`chapters/02-analysis/05-gene-level-ps/02_temporal_vPS_gPS.ipynb` (cells 5-14).

Both notebooks build clusters as connected components over two edge types — colocalisation edges
and shared-lead-variant edges — so the partition is independent of traversal order and reproduces
the notebooks' output exactly. Verified: 20,041 clusters over 70,618 CSs, and every published
statistic to full double precision (see README-r1.md).

Two trait columns are carried throughout:

  raw       `traitFromSourceMappedIds` -- the raw curator mapping the notebooks read. Reproduces
            the published numbers, so it is the control.
  resolved  `diseaseIds` -- the ontology-resolved mapping used by every other analysis in the repo.
            26 raw ids have no row in 25.06 `disease.parquet`; Open Targets had already remapped
            all of them to live MONDO terms here. This is the correct column.

Run scripts that import this from the repository root.
"""

from __future__ import annotations

import collections

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

RELEASE = "data/25.06/output/"
INTERMEDIATE = "data/intermediate_files/"

# Notebook cell 18, in order. First ancestor match wins; unmatched -> "other".
THERAPY_AREA_HIERARCHY = {
    "EFO_0001444": "measurement",
    "MONDO_0045024": "cancer or benign tumor",
    "EFO_0005741": "infectious disease",
    "OTAR_0000009": "injury, poisoning or other complication",
    "OTAR_0000014": "pregnancy or perinatal disease",
    "MONDO_0024458": "disorder of visual system",
    "EFO_0000319": "cardiovascular disease",
    "EFO_0009605": "pancreas disease",
    "EFO_0000540": "immune system disease",
    "EFO_0010282": "gastrointestinal disease",
    "OTAR_0000017": "reproductive system or breast disease",
    "EFO_0010285": "integumentary system disease",
    "EFO_0001379": "endocrine system disease",
    "OTAR_0000010": "respiratory or thoracic disease",
    "EFO_0009690": "urinary system disease",
    "OTAR_0000006": "musculoskeletal or connective tissue disease",
    "MONDO_0021205": "disorder of ear",
    "EFO_0005803": "hematologic disease",
    "EFO_0000618": "nervous system disease",
    "MONDO_0002025": "psychiatric disorder",
    "OTAR_0000020": "nutritional or metabolic disease",
    "OTAR_0000018": "genetic, familial or congenital disease",
    "EFO_0003765": "sign or symptom",
}

RAW_COLUMN = "traitFromSourceMappedIds"
RESOLVED_COLUMN = "diseaseIds"


def load_credible_sets(with_year: bool = False):
    """Load qualifying credible sets in the notebooks' sort order.

    Sorted by p-value ascending then lead-variant PIP descending, matching
    `.sort("pValue", f.desc("locusStatistics.leadVariantPIP"))`. The order only fixes which locus
    seeds each cluster (and hence cluster_id); the partition itself is order-independent.

    Args:
        with_year (bool): join the release study index's publicationDate and derive `year`,
            filling nulls with 2024 as `02_temporal_vPS_gPS.ipynb` cells 10-12 do.

    Returns:
        pd.DataFrame: one row per credible set.
    """
    cs = ds.dataset(INTERMEDIATE + "qualifying_credible_sets", format="parquet").to_table(
        columns={
            "studyLocusId": ds.field("studyLocusId"),
            "studyId": ds.field("studyId"),
            "variantId": ds.field("variantId"),
            "raw": ds.field(RAW_COLUMN),
            "resolved": ds.field(RESOLVED_COLUMN),
            "pValueMantissa": pc.struct_field(ds.field("variantStatistics"), "pValueMantissa"),
            "pValueExponent": pc.struct_field(ds.field("variantStatistics"), "pValueExponent"),
            "leadVariantPIP": pc.struct_field(ds.field("locusStatistics"), "leadVariantPIP"),
        }
    ).to_pandas()

    cs["pValue"] = cs["pValueMantissa"].astype(float) * np.power(10.0, cs["pValueExponent"].astype(float))
    cs = cs.sort_values(["pValue", "leadVariantPIP"], ascending=[True, False], kind="mergesort")

    if with_year:
        study = ds.dataset(RELEASE + "study", format="parquet").to_table(
            columns=["studyId", "publicationDate"]
        ).to_pandas()
        cs = cs.merge(study, on="studyId", how="inner")
        year = pd.to_datetime(cs["publicationDate"], format="%Y-%m-%d", errors="coerce").dt.year
        cs["year"] = year.fillna(2024).astype(int)
        # the merge loses the sort; restore it
        cs = cs.sort_values(["pValue", "leadVariantPIP"], ascending=[True, False], kind="mergesort")

    return cs.reset_index(drop=True)


def load_edges(locus_ids):
    """Load qualifying colocalisation edges with both endpoints inside `locus_ids`.

    Notebook cell 5: coloc unioned with eCAVIAR, kept where h4 >= 0.8 or clpp >= 0.01. Because the
    union fills the missing column with null, that reduces to h4 >= 0.8 for coloc rows and
    clpp >= 0.01 for eCAVIAR rows.

    Args:
        locus_ids (set[str]): study locus ids to restrict both endpoints to.

    Returns:
        list[tuple[str, str]]: undirected edges.
    """
    edges = []
    for path, column, threshold in (
        ("colocalisation_coloc", "h4", 0.8),
        ("colocalisation_ecaviar", "clpp", 0.01),
    ):
        for batch in ds.dataset(RELEASE + path, format="parquet").to_batches(
            columns=["leftStudyLocusId", "rightStudyLocusId", column],
            filter=ds.field(column) >= threshold,
        ):
            left = batch.column("leftStudyLocusId").to_pylist()
            right = batch.column("rightStudyLocusId").to_pylist()
            for l, r in zip(left, right):
                if l in locus_ids and r in locus_ids:
                    edges.append((l, r))
    return edges


def cluster(study_loci, edges):
    """Cluster study loci, reproducing `cluster_lead_variants` (notebook cell 6).

    Args:
        study_loci (list[tuple[str, str]]): (studyLocusId, leadVariantId) in seeding order.
        edges (list[tuple[str, str]]): colocalisation edges.

    Returns:
        list[tuple[str, list[str]]]: (seed studyLocusId, sorted member studyLocusIds) per cluster,
            in cluster_id order.
    """
    locus_to_variant = dict(study_loci)

    variant_to_loci = collections.defaultdict(list)
    for locus_id, variant_id in study_loci:
        variant_to_loci[variant_id].append(locus_id)

    adjacency = collections.defaultdict(set)
    for l, r in edges:
        adjacency[l].add(r)
        adjacency[r].add(l)

    clusters = []
    used = set()
    for seed, _ in study_loci:
        if seed in used:
            continue
        queue = {seed}
        members = set()
        while queue:
            locus_id = queue.pop()
            if locus_id in used:
                continue
            members.add(locus_id)
            used.add(locus_id)
            for neighbour in adjacency.get(locus_id, ()):
                if neighbour not in used:
                    queue.add(neighbour)
            for shared in variant_to_loci.get(locus_to_variant[locus_id], ()):
                if shared not in used:
                    queue.add(shared)
        clusters.append((seed, sorted(members)))
    return clusters


def therapeutic_area_lookup():
    """Build the disease-id to therapeutic-area map of notebook cell 18.

    `disease.parquet` ancestors matched against THERAPY_AREA_HIERARCHY in order, unmatched mapped
    to "other", restricted by a semi-join against the release study index's `diseaseIds`.

    Returns:
        dict[str, str]: disease id -> therapeutic-area id (or "other").
    """
    study_disease_ids = set()
    for batch in ds.dataset(RELEASE + "study", format="parquet").to_batches(columns=["diseaseIds"]):
        for id_list in batch.column("diseaseIds").to_pylist():
            if id_list:
                study_disease_ids.update(id_list)

    disease = ds.dataset(RELEASE + "disease", format="parquet").to_table(
        columns=["id", "ancestors"]
    ).to_pydict()

    lookup = {}
    for disease_id, ancestors in zip(disease["id"], disease["ancestors"]):
        if disease_id not in study_disease_ids:
            continue
        ancestor_set = set(ancestors or ())
        area = next((a for a in THERAPY_AREA_HIERARCHY if a in ancestor_set), None)
        lookup[disease_id] = area if area is not None else "other"
    return lookup


def disease_names():
    """Map disease id to label from `disease.parquet`.

    Returns:
        dict[str, str]: disease id -> name.
    """
    disease = ds.dataset(RELEASE + "disease", format="parquet").to_table(
        columns=["id", "name"]
    ).to_pydict()
    return dict(zip(disease["id"], disease["name"]))


def therapeutic_area_name(area_id):
    """Human-readable label for a therapeutic-area id.

    Args:
        area_id (str): hierarchy id, or the literal "other".

    Returns:
        str: label.
    """
    return THERAPY_AREA_HIERARCHY.get(area_id, "other" if area_id == "other" else area_id)


def cluster_counts(clusters, locus_traits):
    """Per-cluster distinct-disease and distinct-TA counts (notebook cells 13 and 19).

    Args:
        clusters (list): output of `cluster`.
        locus_traits (dict[str, list[str] | None]): studyLocusId -> trait ids for one column.

    Returns:
        tuple[np.ndarray, np.ndarray]: disease counts, TA counts, in cluster_id order.
    """
    lookup = therapeutic_area_lookup()
    diseases, areas = [], []
    for _, members in clusters:
        traits = set()
        for locus_id in members:
            trait_ids = locus_traits.get(locus_id)
            if trait_ids is not None:
                traits.update(trait_ids)
        diseases.append(len(traits))
        areas.append(len({lookup[t] for t in traits if t in lookup}))
    return np.array(diseases), np.array(areas)

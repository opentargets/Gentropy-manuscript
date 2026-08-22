"""Colocalisation clustering of disease credible sets, the basis of the variant pleiotropy score.

Credible sets are grouped into connected components over two edge types: a significant
colocalisation between two sets, and two sets sharing a lead variant. Connected components
are independent of traversal order, so the partition is reproducible. Methods
"Variant-level pleiotropy modelling".

Spark is not needed: the whole problem fits in memory via pyarrow.
"""

import collections

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from manuscript_methods import paper

COLOC_H4 = 0.8
ECAVIAR_CLPP = 0.01


def load_credible_sets(path: str = None) -> pd.DataFrame:
    """Qualifying disease credible sets, sorted by p-value then lead-variant PIP.

    The sort decides which locus seeds each cluster, and hence both the cluster numbering and
    which lead variant represents the cluster downstream.

    Ranking on `variantStatistics.chi2Stat` descending was tried on 2026-08-22 and reverted the
    same day. Chi-square is monotone in the p-value at one degree of freedom, and here it is
    computed *from* the stored mantissa and exponent, so it carries the same information and cannot
    break a tie the stored p-value does not break: cluster 26's three most significant credible
    sets all store `1.0e-323` and all map to `chi2Stat = 1479.141`, so the representative stayed
    `19_44888997_C_T` either way. The only effect was to permute 22 of the 20,041 `cluster_id`
    values, which moved the ST15 sheet for no analytical gain. See
    `chapters/02-analysis-main/README.md`.

    `chi2Stat` is kept on the frame — it is non-null and strictly positive for every qualifying
    credible set, in every project and both `hasSumstats` strata — so the alternative ordering
    stays computable.
    """
    path = path or paper.derived("qualifying_credible_sets")
    table = ds.dataset(path, format="parquet").to_table(
        columns={
            "studyLocusId": ds.field("studyLocusId"),
            "studyId": ds.field("studyId"),
            "variantId": ds.field("variantId"),
            "diseaseIds": ds.field("diseaseIds"),
            "traitFromSourceMappedIds": ds.field("traitFromSourceMappedIds"),
            "chi2Stat": pc.struct_field(ds.field("variantStatistics"), "chi2Stat"),
            "pValueMantissa": pc.struct_field(ds.field("variantStatistics"), "pValueMantissa"),
            "pValueExponent": pc.struct_field(ds.field("variantStatistics"), "pValueExponent"),
            "leadVariantPIP": pc.struct_field(ds.field("locusStatistics"), "leadVariantPIP"),
        }
    )
    cs = table.to_pandas()
    cs["pValue"] = cs["pValueMantissa"].astype(float) * np.power(10.0, cs["pValueExponent"].astype(float))
    if cs["chi2Stat"].isna().any():
        raise ValueError(f"chi2Stat is null for {int(cs['chi2Stat'].isna().sum())} credible sets")
    cs = cs.sort_values(["pValue", "leadVariantPIP"], ascending=[True, False], kind="mergesort")
    return cs.reset_index(drop=True)


def load_edges(locus_ids: set) -> list:
    """Significant colocalisation pairs with both credible sets inside `locus_ids`."""
    edges = []
    for name, column, threshold in (
        ("colocalisation_coloc", "h4", COLOC_H4),
        ("colocalisation_ecaviar", "clpp", ECAVIAR_CLPP),
    ):
        dataset = ds.dataset(paper.release(name), format="parquet")
        for batch in dataset.to_batches(
            columns=["leftStudyLocusId", "rightStudyLocusId", column], filter=ds.field(column) >= threshold
        ):
            left = batch.column("leftStudyLocusId").to_pylist()
            right = batch.column("rightStudyLocusId").to_pylist()
            edges += [(a, b) for a, b in zip(left, right) if a in locus_ids and b in locus_ids]
    return edges


def cluster(study_loci: list, edges: list) -> list:
    """Connected components over colocalisation edges and shared lead variants.

    Args:
        study_loci: (studyLocusId, variantId) pairs in seeding order.
        edges: colocalisation pairs.

    Returns:
        list of (seed studyLocusId, sorted member studyLocusIds), in cluster order.
    """
    locus_variant = dict(study_loci)

    variant_loci = collections.defaultdict(list)
    for locus_id, variant_id in study_loci:
        variant_loci[variant_id].append(locus_id)

    adjacency = collections.defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    clusters, used = [], set()
    for seed, _ in study_loci:
        if seed in used:
            continue
        queue, members = {seed}, set()
        while queue:
            locus_id = queue.pop()
            if locus_id in used:
                continue
            members.add(locus_id)
            used.add(locus_id)
            queue |= {n for n in adjacency.get(locus_id, ()) if n not in used}
            queue |= {n for n in variant_loci.get(locus_variant[locus_id], ()) if n not in used}
        clusters.append((seed, sorted(members)))
    return clusters


def therapeutic_area_lookup(column: str = "primaryTherapeuticArea") -> dict:
    """Disease id to therapeutic area, restricted to terms used by the release study index.

    Defaults to the hierarchy order published as Supplementary Table 9, the same order the
    gene-level analysis uses. The published cluster-level therapeutic-area counts were computed
    under the legacy order instead; pass `primaryTherapeuticAreaLegacy` to recover them. See
    `01-data-preparation/03_therapeutic_areas`.
    """
    efo_ta = ds.dataset(paper.derived("efo_therapeutic_area"), format="parquet").to_table().to_pydict()
    return dict(zip(efo_ta["id"], efo_ta[column]))


def disease_names() -> dict:
    """Disease id to label."""
    disease = (
        ds.dataset(paper.release("disease") + "/disease.parquet", format="parquet")
        .to_table(columns=["id", "name"])
        .to_pydict()
    )
    return dict(zip(disease["id"], disease["name"]))


def cluster_table(cs: pd.DataFrame, clusters: list, trait_column: str = "diseaseIds") -> pd.DataFrame:
    """Per-cluster distinct-disease and distinct-therapeutic-area counts.

    `uniqueDiseases` is the variant pleiotropy score (vPS) of the cluster.
    """
    lookup = therapeutic_area_lookup()
    traits_by_locus = dict(zip(cs["studyLocusId"], cs[trait_column]))
    variant_by_locus = dict(zip(cs["studyLocusId"], cs["variantId"]))

    rows = []
    for cluster_id, (seed, members) in enumerate(clusters):
        traits = set()
        for locus_id in members:
            ids = traits_by_locus.get(locus_id)
            if ids is not None:
                traits.update(ids)
        areas = {lookup[t] for t in traits if t in lookup}
        rows.append(
            {
                "cluster_id": cluster_id,
                "leadStudyLocusId": seed,
                "leadVariantId": variant_by_locus[seed],
                "clusterSize": len(members),
                "uniqueLeadVariants": len({variant_by_locus[m] for m in members}),
                "uniqueDiseases": len(traits),
                "uniqueTherapeuticAreas": len(areas),
            }
        )
    return pd.DataFrame(rows)


def membership_table(cs: pd.DataFrame, clusters: list, trait_column: str = "diseaseIds") -> pd.DataFrame:
    """One row per cluster and disease, for Supplementary Table 15."""
    lookup = therapeutic_area_lookup()
    names = disease_names()
    traits_by_locus = dict(zip(cs["studyLocusId"], cs[trait_column]))
    variant_by_locus = dict(zip(cs["studyLocusId"], cs["variantId"]))

    rows = []
    for cluster_id, (_, members) in enumerate(clusters):
        variants = sorted({variant_by_locus[m] for m in members})
        traits = set()
        for locus_id in members:
            ids = traits_by_locus.get(locus_id)
            if ids is not None:
                traits.update(ids)
        for disease_id in sorted(traits):
            area = lookup.get(disease_id, "other")
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "leadVariants": ";".join(variants),
                    "diseaseId": disease_id,
                    "diseaseName": names.get(disease_id),
                    "therapeuticArea": paper.THERAPEUTIC_AREAS.get(area, "other"),
                    "vPS": len(traits),
                }
            )
    return pd.DataFrame(rows)

"""Shared definitions for the effective-independent-traits analysis (R2-MJ-3/7b/8/12, R1-mn-8b, R1-MJ-2).

One place for the Li & Ji (2005) estimator and the gene-trait loaders, used by
`01_metrics_and_gate.ipynb`. Notebook 02 consumes this notebook's exported per-gene table
(`eit_gene_metrics-r1.csv`) rather than recomputing the metrics, and imports its enrichment statistics
from `../or10-optimism-validation/or10_stats.py`.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

INTERMEDIATE = "../../../data/intermediate_files/"

GENE_TABLE = "genes_therapeutic_areas.csv"
DISEASE_L2G = "l2g_diseases_full-r1.csv"
MEASUREMENT_L2G = "l2g_measurements_full-r1.csv"
RG_MATRIX = "canonical_pairwise_table/rg_processed.parquet"


def meff_li_ji(sub: np.ndarray) -> float:
    """Li & Ji (2005) effective number of independent tests for a correlation submatrix.

    lam  = |eigenvalues|            (absolute values, so no PSD repair is required)
    meff = sum( (lam >= 1) + (lam - floor(lam)) )

    Exact for independent traits: k orthogonal traits give exactly k.

    Two caveats, both properties of the published estimator rather than of this implementation, and
    both measured in `01_metrics_and_gate.ipynb` (exported to `eit_estimator_robustness-r1.csv`):

    - the summand is **discontinuous at every integer >= 2**, so exactly degenerate input is decided by
      floating-point rounding (an all-ones k x k block returns 1.0 for k = 2, 6-10, 12 but 2.0 for
      k = 3, 4, 5, 11);
    - for any pairwise correlation below 1 the estimator **floors a duplicate cluster near 2, not 1**,
      so it over-states independence for near-duplicate trait sets and the redundancy it reports is a
      lower bound.

    The brief's "k identical traits give 1" therefore holds only in the exact-arithmetic limit.
    """
    if sub.shape[0] == 1:
        return 1.0
    lam = np.abs(np.linalg.eigvalsh(sub))
    return float(((lam >= 1).astype(float) + (lam - np.floor(lam))).sum())


def load_rg(base: str = INTERMEDIATE) -> pd.DataFrame:
    """Genetic-correlation matrix S over representative studies, indexed by EFO/MONDO/HP/OBA id."""
    return pd.read_parquet(base + RG_MATRIX)


def load_gene_table(base: str = INTERMEDIATE) -> pd.DataFrame:
    """Load the published per-gene table.

    `uniqueDiseases` is gPS and `uniqueTherapeuticAreas` is gps_TA. Note that gps_TA is *not* a
    one-therapeutic-area-per-disease count — it is the union, across the gene's contributing studies,
    of each study's set of top-of-ontology areas. See deviation 5 in the README.
    """
    return pd.read_csv(base + GENE_TABLE)


def gene_trait_pairs(filename: str, genes: set[str] | None = None, base: str = INTERMEDIATE) -> pd.DataFrame:
    """Long gene -> trait table from an L2G association export.

    `diseaseIds` is stored as a Python list-repr string, so it is parsed with `ast.literal_eval`
    rather than split on commas (a plain string split inflates the vocabulary; see the published
    Figure 1 panel c bug).
    """
    df = pd.read_csv(base + filename, usecols=["geneId", "diseaseIds"])
    if genes is not None:
        df = df[df["geneId"].isin(genes)]
    df = df.assign(traitId=df["diseaseIds"].map(ast.literal_eval)).explode("traitId")
    return df[["geneId", "traitId"]].drop_duplicates().reset_index(drop=True)


def meff_per_gene(overlap_sets: pd.Series, matrix: np.ndarray) -> pd.Series:
    """Meff for each gene, memoised on the exact set of matrix indices."""
    cache: dict[tuple[int, ...], float] = {}
    out = {}
    for gene_id, idx in overlap_sets.items():
        if idx not in cache:
            cache[idx] = meff_li_ji(matrix[np.ix_(list(idx), list(idx))])
        out[gene_id] = cache[idx]
    return pd.Series(out, name="meff")

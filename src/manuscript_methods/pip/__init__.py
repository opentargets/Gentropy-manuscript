"""Posterior Inclusion Probability (PIP) methods."""

from pyspark.sql import Column
from pyspark.sql import functions as f


def extract_pip_from_locus(variant_col: Column, locus: Column) -> Column:
    """Extract Posterior probability from variant from locus.

    In the case the lead variant is not present in the locus, None is returned.
    """
    lead_variant_stats = f.filter(locus, lambda v: v.getField("variantId") == variant_col)
    return (
        f.when(
            f.size(lead_variant_stats) == 1,
            lead_variant_stats.getItem(0).getField("posteriorProbability"),
        )
        .otherwise(None)
        .alias("posteriorProbability")
    )

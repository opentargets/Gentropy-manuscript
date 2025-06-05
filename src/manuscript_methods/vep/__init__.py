"""VEP related functions."""

from pyspark.sql import Column
from pyspark.sql import functions as f


def vep_variant_effect(c: Column) -> Column:
    """Extract VEP variant effect."""

    def extract_fields(ve: Column) -> Column:
        return f.struct(
            ve.getField("assessment").alias("assessment"),
            ve.getField("normalisedScore").alias("normalisedScore"),
            ve.getField("targetId").alias("targetId"),
        )

    return f.transform(f.filter(c, lambda ve: ve.getField("method") == f.lit("VEP")), extract_fields).getItem(0)

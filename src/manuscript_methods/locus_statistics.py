"""Locus object statistics."""

from __future__ import annotations

from typing import NamedTuple

from gentropy.common.utils import extract_chromosome, extract_position
from pyspark.sql import Column
from pyspark.sql import functions as f
from pyspark.sql import types as t

from manuscript_methods.variant_type import Variant


def extract_ref(variant_id: Column) -> Column:
    """Extract ref from variant ID.

    This function extracts the ref from a variant ID. The variantId is expected to be in the format `chromosome_position_ref_alt`.
    If the variant ID starts with "OT_VAR_", it returns None, as these IDs do not contain ref and alt information.

    Args:
        variant_id (Column): Variant ID

    Returns:
        Column: Ref

    Examples:
        >>> d = [("OT_VAR_1_12343",),("chr1_12345_AT_A",),("15_KI270850v1_alt_48777_C_T",),]
        >>> df = spark.createDataFrame(d).toDF("variantId")
        >>> df.withColumn("ref", extract_ref(f.col("variantId"))).show(truncate=False)
        +---------------------------+----+
        |variantId                  |ref |
        +---------------------------+----+
        |OT_VAR_1_12343             |NULL|
        |chr1_12345_AT_A            |AT  |
        |15_KI270850v1_alt_48777_C_T|C   |
        +---------------------------+----+
        <BLANKLINE>

    """
    return (
        f.when(variant_id.startswith("OT_VAR_"), f.lit(None))
        .otherwise(f.regexp_extract(variant_id, r"^.*_\d+_([ATGC]+)_([ATGC]+)$", 1))
        .alias("ref")
    )


def extract_alt(variant_id: Column) -> Column:
    """Extract alt from variant ID.

    This function extracts the alt from a variant ID. The variantId is expected to be in the format `chromosome_position_ref_alt`.
    If the variant ID starts with "OT_VAR_", it returns None, as these IDs do not contain ref and alt information.

    Args:
        variant_id (Column): Variant ID

    Returns:
        Column: Alt

    Examples:
        >>> d = [("OT_VAR_1_12343",),("chr1_12345_A_AT",),("15_KI270850v1_alt_48777_C_T",),]
        >>> df = spark.createDataFrame(d).toDF("variantId")
        >>> df.withColumn("alt", extract_alt(f.col("variantId"))).show(truncate=False)
        +---------------------------+----+
        |variantId                  |alt |
        +---------------------------+----+
        |OT_VAR_1_12343             |NULL|
        |chr1_12345_A_AT            |AT  |
        |15_KI270850v1_alt_48777_C_T|T   |
        +---------------------------+----+
        <BLANKLINE>

    """
    return (
        f.when(variant_id.startswith("OT_VAR_"), f.lit(None))
        .otherwise(f.regexp_extract(variant_id, r"^.*_\d+_([ATGC]+)_([ATGC]+)$", 2))
        .alias("alt")
    )


class LocusRanges(NamedTuple):
    """Named tuple to hold locus range information."""

    locus_start: Column
    locus_end: Column

    @classmethod
    def from_locus(cls, locus: Column) -> LocusRanges:
        """Extract the start and end of the locus from a given column.

        Args:
            locus (Column): The column containing locus information.

        Returns:
            LocusRanges: A named tuple with locus start and end columns.

        """
        # NOTE!  Calculate variant start and end positions:
        # The locus object does not have the `position` and `chromosome`, `ref`, `alt`
        # required to calculate the variant start and end, so
        # we need to extract them from the `variantId`
        # In case of OT_VAR identifieres, we can not extract the position.
        variant_bounds = f.transform(
            locus,
            lambda v: Variant.compute(
                extract_chromosome(v.getField("variantId")),
                extract_position(v.getField("variantId")).cast(t.IntegerType()),
                extract_ref(v.getField("variantId")),
                extract_alt(v.getField("variantId")),
            ).col,
        )
        locus_start = f.reduce(
            variant_bounds,
            variant_bounds.getItem(0).getField("start"),
            lambda acc, v: f.least(acc, v.getField("start")),
        ).alias("locusStart")
        locus_end = f.reduce(
            variant_bounds,
            variant_bounds.getItem(0).getField("end"),
            lambda acc, v: f.greatest(acc, v.getField("start")),
        ).alias("locusEnd")
        return cls(locus_start, locus_end)


class LocusStatistics:
    name: str = "locusStatistics"
    schema: str = "struct<locusId: STRING, locusSize: INT, locusStart: INT, locusEnd: INT, leadVariantPIP: FLOAT>"

    def __init__(self, col: Column | None = None):
        """Initialize LocusStatistics with an optional column.

        Args:
            col (f.Column, optional): Optional column to initialize the locus statistics.

        """
        self.col = col.alias(self.name) if col is not None else f.col(self.name)

    @property
    def lead_variant_pip(self) -> Column:
        """Get the lead variant PIP from the locus statistics."""
        return self.col.getField("leadVariantPIP")

    @classmethod
    def extract_pip_from_locus(cls, variant_col: Column, locus: Column) -> Column:
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
            .alias("leadVariantPIP")
        )

    @classmethod
    def compute(cls, locus: Column, lead_variant: Column) -> LocusStatistics:
        """Compute locus statistics based on the provided locus and lead variant.

        Args:
            locus (Column): The column containing locus information.
            lead_variant (Column): The column containing lead variant information.

        Returns:
            LocusStatistics: An instance of LocusStatistics with computed values.

        """
        ranges = LocusRanges.from_locus(locus)
        return LocusStatistics(
            f.struct(
                f.size(locus).alias("locusSize"),
                (ranges.locus_end - ranges.locus_start).alias("locusLength"),
                ranges.locus_start.alias("locusStart"),
                ranges.locus_end.alias("locusEnd"),
                cls.extract_pip_from_locus(lead_variant, locus).alias("leadVariantPIP"),
            )
        )

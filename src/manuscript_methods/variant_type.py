from __future__ import annotations
from pyspark.sql import Column
from pyspark.sql import functions as f
from enum import Enum


class VariantType(str, Enum):
    """Enum representing different types of variants."""

    SNV = "SNV"
    INSERTION = "INS"
    DELETION = "DEL"


class Variant:
    """Class representing a variant type."""

    name = "variantType"
    schema = "STRUCT<type: STRING, start: INT, end: INT, ref: STRING, alt: STRING>"

    def __init__(self, col: Column | None = None) -> None:
        """Initialize the Variant class with a column."""
        self.col = col.alias(self.name) if col is not None else f.col(self.name)

    @staticmethod
    def get_variant_type(ref: Column, alt: Column) -> Column:
        """Get the variant type from a Variant instance."""
        expr = (
            f.when((f.length(alt) > f.length(ref)), VariantType.INSERTION)
            .when((f.length(alt) < f.length(ref)), VariantType.DELETION)
            .when((f.length(alt) == 1 & f.length(ref) == 1), VariantType.SNV)
        )
        return expr.alias("type")

    @staticmethod
    def get_svlen(ref: Column, alt: Column) -> Column:
        """Get the indel length of the variant.

        Note:
        ----
        The length is positive for insertions and negative for deletions.

        """
        return f.length(alt) - f.length(ref)

    @classmethod
    def get_variant_length(cls, ref: Column, alt: Column) -> Column:
        """Get the length of the variant."""
        return f.when(f.length(alt) != f.length(ref), cls.get_svlen(ref, alt)).otherwise(f.lit(0)).alias("length")

    @staticmethod
    def get_variant_end(pos: Column, length: Column) -> Column:
        """Get the end position of the variant."""
        return pos + length

    @classmethod
    def compute(cls, pos: Column, ref: Column, alt: Column) -> Variant:
        """Compute the variant type based on position, reference, and alternate alleles."""
        return Variant(
            f.struct(
                cls.get_variant_type(ref, alt).alias("type"),
                pos.alias("start"),
                (pos + f.length(ref) - 1).alias("end"),
                ref.alias("ref"),
                alt.alias("alt"),
            )
        )

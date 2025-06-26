from __future__ import annotations

from enum import Enum

from pyspark.sql import Column
from pyspark.sql import functions as f


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
            .when(((f.length(alt) == 1) & (f.length(ref) == 1)), VariantType.SNV)
        )
        return expr.alias("type")

    @classmethod
    def get_variant_len(cls, ref: Column, alt: Column) -> Column:
        """Get the indel length of the variant.

        Note:
        ----
        Effective length is defined as the absolute difference between the lengths of the reference and alternate alleles.

        """
        return f.abs(f.length(alt) - f.length(ref)).alias("length")

    @classmethod
    def get_variant_end(cls, pos: Column, ref: Column, alt: Column) -> Column:
        """Get the end position of the variant."""
        return pos + cls.get_variant_len(ref, alt).alias("end")

    @classmethod
    def compute(cls, chr: Column, pos: Column, ref: Column, alt: Column) -> Variant:
        """Compute the variant type based on position, reference, and alternate alleles."""
        return Variant(
            f.struct(
                chr,
                pos.alias("start"),
                cls.get_variant_end(pos, ref, alt).alias("end"),
                cls.get_variant_type(ref, alt).alias("type"),
                ref.alias("ref"),
                alt.alias("alt"),
                cls.get_variant_len(ref, alt).alias("length"),
            )
        )

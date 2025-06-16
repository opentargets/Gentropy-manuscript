"""VEP related functions."""

from collections.abc import Callable
from enum import Enum

from pyspark.sql import Column
from pyspark.sql import functions as f


class SingleVariantEffectMethod(str, Enum):
    """Enum for VEP variant effect methods."""

    ALPHA_MISSENSE = "AlphaMissense"
    FOLDX = "FoldX"
    GERP = "GERP"
    LOFTEE = "LOFTEE"
    SIFT = "SIFT"
    VEP = "VEP"


class SingleVariantEffect:
    """Class to extract VEP variant effect from a column."""

    schema = "STRUCT<method: STRING, assessment: STRING, score: FLOAT, assessmentFlag: STRING, targetId: STRING, normalisedScore: DOUBLE>"

    def __init__(self, col: Column | None = None):
        """Initialize with a column."""
        self.name = "singleVariantEffect"
        self.col = col.alias(self.name) if col is not None else f.col(self.name)

    @property
    def method(self) -> Column:
        """Get the method field."""
        return self.col.getField("method")

    @property
    def assessment(self) -> Column:
        """Get the assessment field."""
        return self.col.getField("assessment")

    @property
    def normalised_score(self) -> Column:
        """Get the normalised score field."""
        return self.col.getField("normalisedScore")

    @property
    def target_id(self) -> Column:
        """Get the target ID field."""
        return self.col.getField("targetId")


class VariantEffect:
    """Class representing a list of variant effects."""

    schema = f"ARRAY<{SingleVariantEffect.schema}>"

    def __init__(self, col: Column | None = None):
        """Initialize with a column."""
        self.name = "variantEffect"
        self.col = col.alias(self.name) if col is not None else f.col(self.name)

    def filter_effect_by_method(self, method: SingleVariantEffectMethod) -> SingleVariantEffect:
        """Filter variant effects by method."""
        _filter: Callable[[Column], Column]
        _filter = lambda ve: ve.getField("method") == f.lit(method.value)
        expr = f.filter(self.col, _filter)
        return SingleVariantEffect(expr)

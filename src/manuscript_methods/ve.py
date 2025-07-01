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
    """Class to extract VEP variant effect from a column.

    Examples
    --------
    >>> data = [(("VEP", "intron_variant,", 0.1, None, "ENSG00000123456", 0.1),)]
    >>> schema = f"variantEffect: {SingleVariantEffect.schema}"
    >>> df = spark.createDataFrame(data, schema)
    >>> df.show(truncate=False)
    +-------------------------------------------------------+
    |variantEffect                                          |
    +-------------------------------------------------------+
    |{VEP, intron_variant,, 0.1, NULL, ENSG00000123456, 0.1}|
    +-------------------------------------------------------+
    <BLANKLINE>
    >>> ve = SingleVariantEffect(df.variantEffect)
    >>> df.select(ve.method, ve.normalised_score, ve.assessment, ve.target_id).show(truncate=False)
    +------+---------------+---------------+---------------+
    |method|normalisedScore|assessment     |targetId       |
    +------+---------------+---------------+---------------+
    |VEP   |0.1            |intron_variant,|ENSG00000123456|
    +------+---------------+---------------+---------------+
    <BLANKLINE>

    """

    schema = "STRUCT<method: STRING, assessment: STRING, score: FLOAT, assessmentFlag: STRING, targetId: STRING, normalisedScore: DOUBLE>"

    def __init__(self, col: Column | None = None):
        """Initialize with a column."""
        self.name = "singleVariantEffect"
        self.col = col.alias(self.name) if col is not None else f.col(self.name)

    @property
    def method(self) -> Column:
        """Get the method field."""
        return self.col.getField("method").alias("method")

    @property
    def assessment(self) -> Column:
        """Get the assessment field."""
        return self.col.getField("assessment").alias("assessment")

    @property
    def normalised_score(self) -> Column:
        """Get the normalised score field."""
        return self.col.getField("normalisedScore").alias("normalisedScore")

    @property
    def target_id(self) -> Column:
        """Get the target ID field."""
        return self.col.getField("targetId").alias("targetId")


class VariantEffect:
    """Class representing a list of variant effects.

    Examples
    --------
    >>> x1 = ("VEP", "intron_variant,", 0.1, None, "ENSG00000123456", 0.1)
    >>> x2 = ("SIFT", "missense_variant", 0.05, None, "ENSG00000123456", 0.05)
    >>> data = [([x1, x2],),]
    >>> schema = f"variantEffects: {VariantEffect.schema}"
    >>> df = spark.createDataFrame(data, schema)
    >>> ve = VariantEffect(df.variantEffects)
    >>> df.select(ve.filter_effect_by_method(SingleVariantEffectMethod.VEP).col).show(truncate=False)
    +---------------------------------------------------------+
    |singleVariantEffect                                      |
    +---------------------------------------------------------+
    |[{VEP, intron_variant,, 0.1, NULL, ENSG00000123456, 0.1}]|
    +---------------------------------------------------------+
    <BLANKLINE>

    """

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

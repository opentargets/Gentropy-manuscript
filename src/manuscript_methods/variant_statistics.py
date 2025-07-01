"""Calculate variance explained."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from pyspark.sql import Column
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.pandas.functions import PandasUDFType, pandas_udf
from scipy.stats import chi2


@pandas_udf(returnType=t.DoubleType(), functionType=PandasUDFType.SCALAR)
def chi2_inverse_survival_function(x: pd.Series) -> pd.Series:
    """Calculate the inverse survival function of the chi2 distribution with 1 degree of freedom.

    Args:
        x (pd.Series): A pandas Series containing p-values (between 0 and 1).

    Returns:
        pd.Series: A pandas Series containing the chi2 statistic corresponding to the input p-values.

    Get the chi2 statistic for a given p-value (x).

    Examples:
    --------
    >>> data = [(0.1,), (0.05,), (0.001,)]
    >>> schema = "pValue Float"
    >>> df = spark.createDataFrame(data, schema=schema)
    >>> df.show()
    +------+
    |pValue|
    +------+
    |   0.1|
    |  0.05|
    | 0.001|
    +------+
    <BLANKLINE>

    >>> import pyspark.sql.functions as f
    >>> chi2 = f.round(chi2_inverse_survival_function("pValue"), 2).alias("chi2_stat")
    >>> df.select("pValue", chi2).show()
    +------+---------+
    |pValue|chi2_stat|
    +------+---------+
    |   0.1|     2.71|
    |  0.05|     3.84|
    | 0.001|    10.83|
    +------+---------+
    <BLANKLINE>

    """
    return pd.Series(chi2.isf(x, df=1).astype(np.float64))


class PValueComponents(NamedTuple):
    """Components of a p-value for lead variant statistics."""

    p_value_mantissa: Column
    p_value_exponent: Column

    def neglog(self) -> Column:
        """Calculate the negative logarithm of the p-value."""
        return -1 * (f.log10(self.p_value_mantissa) + self.p_value_exponent).alias("neglogPValue")

    def chi2(self) -> Column:
        """Calculate chi2 from p-value.

        This function calculates the chi2 value from the p-value mantissa and exponent.
        In case the p-value is very small (exponent < -300), it uses an approximation based on a linear regression model.
        The approximation is based on the formula: -5.367 * neglog_pval + 4.596, where neglog_pval is the negative log10 of the p-value mantissa.

        Returns:
            Column: Chi2 value (float)

        Examples:
        --------
        >>> data = [(5.0, -8), (9.0, -300), (9.0, -301)]
        >>> schema = "pValueMantissa FLOAT, pValueExponent INT"
        >>> df = spark.createDataFrame(data, schema)
        >>> df.show()
        +--------------+--------------+
        |pValueMantissa|pValueExponent|
        +--------------+--------------+
        |           5.0|            -8|
        |           9.0|          -300|
        |           9.0|          -301|
        +--------------+--------------+
        <BLANKLINE>

        >>> mantissa = f.col("pValueMantissa")
        >>> exponent = f.col("pValueExponent")
        >>> components = PValueComponents(mantissa, exponent)
        >>> chi2 = f.round(components.chi2(), 2).alias("chi2")
        >>> df2 = df.select(mantissa, exponent, chi2)
        >>> df2.show()
        +--------------+--------------+-------+
        |pValueMantissa|pValueExponent|   chi2|
        +--------------+--------------+-------+
        |           5.0|            -8|  29.72|
        |           9.0|          -300|1369.48|
        |           9.0|          -301|1373.64|
        +--------------+--------------+-------+
        <BLANKLINE>

        """
        PVAL_EXP_THRESHOLD = f.lit(-300)
        APPROX_INTERCEPT = f.lit(-5.367)
        APPROX_COEF = f.lit(4.596)
        neglog_pval = self.neglog()
        p_value = self.p_value_mantissa * f.pow(10, self.p_value_exponent)
        neglog_approx = (neglog_pval * APPROX_COEF + APPROX_INTERCEPT).cast(t.DoubleType())

        return (
            f.when(self.p_value_exponent < PVAL_EXP_THRESHOLD, neglog_approx)
            .otherwise(chi2_inverse_survival_function(p_value))
            .alias("chi2Stat")
        )


class VariantStatistics:
    """Calculate variant statistics."""

    name = "variantStats"
    schema = "struct<chi2Stat:double , pValueMantissa:double, pValueExponent: int, approxVarianceExplained: double>"

    def __init__(self, col: Column | None = None):
        """Initialize LeadVariantStatistics with an optional column.

        Args:
            col (Column, optional): A Spark SQL Column representing the lead variant statistics. If None,
                a default column with the name 'variantStats' will be created. Defaults to None.

        Examples:
        --------
        >>> data = [((29.717, 5.0, -8, 0.03),),]
        >>> print(VariantStatistics.schema)
        struct<chi2Stat:double , pValueMantissa:double, pValueExponent: int, approxVarianceExplained: double>
        >>> df = spark.createDataFrame(data, schema=f"variantStats: {VariantStatistics.schema}")
        >>> df.show(truncate=False)
        +-----------------------+
        |variantStats           |
        +-----------------------+
        |{29.717, 5.0, -8, 0.03}|
        +-----------------------+
        <BLANKLINE>

        """
        self.col = col.alias(self.name) if col is not None else f.col(self.name)

    @property
    def chi2_stat(self) -> Column:
        """Get the chi2 statistic from the variant statistics."""
        return self.col.getField("chi2Stat").alias("chi2Stat")

    @staticmethod
    def approx_variance_explained(chi2_stat: Column, n_samples: Column) -> Column:
        """Calculate the variance explained by the lead variant."""
        return (chi2_stat / n_samples).alias("ApproximatedVarianceExplained")

    @classmethod
    def compute(cls, p_value_components: PValueComponents, n_samples: Column) -> VariantStatistics:
        """Compute the variant statistics from p-value components and number of samples.

        Args:
            p_value_components (PValueComponents): Components of the p-value (mantissa and exponent).
            n_samples (Column): Number of samples in the study.

        Returns:
            Column: A column containing the lead variant statistics, including chi2 statistic, p-value components

        Examples:
        --------
        >>> data = [(5.0, -8),]
        >>> schema = "pValueMantissa DOUBLE, pValueExponent INT"
        >>> df = spark.createDataFrame(data, schema)
        >>> p_value_components = PValueComponents(f.col("pValueMantissa"), f.col("pValueExponent"))
        >>> n_samples = f.lit(1000)
        >>> variant_stats = VariantStatistics.compute(p_value_components, n_samples).col
        >>> df = df.withColumn("variantStats", variant_stats).select("variantStats.*")
        >>> cols = [f.round(f.col(c), 3).alias(c) for c in df.columns]
        >>> df.select(*cols).show(truncate=False)
        +--------+--------------+--------------+-----------------------------+
        |chi2Stat|pValueMantissa|pValueExponent|ApproximatedVarianceExplained|
        +--------+--------------+--------------+-----------------------------+
        |29.717  |5.0           |-8            |0.03                         |
        +--------+--------------+--------------+-----------------------------+
        <BLANKLINE>

        """
        chi2_stat = p_value_components.chi2()
        approx_variance_explained = cls.approx_variance_explained(chi2_stat, n_samples)
        expr = f.struct(
            chi2_stat,
            p_value_components.p_value_mantissa,
            p_value_components.p_value_exponent,
            approx_variance_explained,
        ).alias("leadVariantStats")

        return cls(expr)

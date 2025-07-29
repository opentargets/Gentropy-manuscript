"""Resources to classify studyIds."""

from collections.abc import Callable
from enum import Enum
from typing import NamedTuple

from gentropy import StudyIndex, StudyLocus
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f


class DataSource(NamedTuple):
    """Resource class to hold resource information."""

    name: str
    filter_expr: Callable[[], Column]


class DataSourceType(Enum):
    """Resource type enum."""

    GWAS_CATALOG = DataSource(
        name="GWAS Catalog",
        filter_expr=lambda: (f.col("studyId").startswith(f.lit("GCST"))),
    )
    EQLT_CATALOGUE = DataSource(
        name="eQTL Catalogue",
        filter_expr=lambda: (
            (~f.col("studyId").startswith(f.lit("GCST")))
            & (~f.col("studyId").startswith(f.lit("FINNGEN")))
            & (~f.col("studyId").startswith(f.lit("UKB")))
        ),
    )
    UKB_PPP_EUR = DataSource(
        name="UK Biobank PPP EUR",
        filter_expr=lambda: (f.col("studyId").startswith(f.lit("UKB"))),
    )
    FINNGEN = DataSource(
        name="FinnGen",
        filter_expr=lambda: (f.col("studyId").startswith(f.lit("FINNGEN"))),
    )


def describe_datasource(df: DataFrame) -> DataFrame:
    """Describe the resource.

    Args:
        df (DataFrame): DataFrame containing studyId or studyLocusId.

    Returns:
        DataFrame: DataFrame with dataSourceId and count.

    Examples:
        >>> data = [("GCST000001",), ("GCST000002",), ("FINNGEN_R12-0001",)]
        >>> schema = "studyId STRING"
        >>> df = spark.createDataFrame(data, schema)
        >>> describe_datasource(df).show()
        +------------+-----+
        |dataSourceId|count|
        +------------+-----+
        |GWAS Catalog|    2|
        |     FinnGen|    1|
        +------------+-----+
        <BLANKLINE>

    """
    return classify_by_datasource(df).groupBy("dataSourceId").count()


def classify_by_datasource(df: DataFrame) -> DataFrame:
    """Classify the DataFrame by resource.

    Args:
        df (DataFrame): DataFrame containing studyId or studyLocusId.

    Returns:
        DataFrame: DataFrame with dataSourceId.

    """
    expr = f.when(f.lit(False), f.lit(None).cast("string"))
    for x in [
        DataSourceType.GWAS_CATALOG,
        DataSourceType.EQLT_CATALOGUE,
        DataSourceType.UKB_PPP_EUR,
        DataSourceType.FINNGEN,
    ]:
        expr = expr.when(
            x.value.filter_expr(),
            f.lit(x.value.name),
        )
    expr = expr.alias("dataSourceId")
    return df.withColumn("dataSourceId", expr)

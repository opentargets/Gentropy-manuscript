"""Resources to classify studyIds."""

from collections.abc import Callable
from enum import Enum
from typing import NamedTuple

from gentropy import StudyIndex, StudyLocus
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f


class Resource(NamedTuple):
    """Resource class to hold resource information."""

    name: str
    filter_expr: Callable[[], Column]


class ResourceType(Enum):
    """Resource type enum."""

    GWAS_CATALOG = Resource(
        name="GWAS Catalog",
        filter_expr=lambda: (f.col("studyId").startswith(f.lit("GCST"))),
    )
    EQLT_CATALOGUE = Resource(
        name="eQTL Catalogue",
        filter_expr=lambda: (
            (~f.col("studyId").startswith(f.lit("GCST")))
            & (~f.col("studyId").startswith(f.lit("FINNGEN")))
            & (~f.col("studyId").startswith(f.lit("UKB")))
        ),
    )
    UKB_PPP_EUR = Resource(
        name="UK Biobank PPP EUR",
        filter_expr=lambda: (f.col("studyId").startswith(f.lit("UKB"))),
    )
    FINNGEN = Resource(
        name="FinnGen",
        filter_expr=lambda: (f.col("studyId").startswith(f.lit("FINNGEN"))),
    )


def describe_resource(df: DataFrame) -> DataFrame:
    """Describe the resource.

    Args:
        df (DataFrame): DataFrame containing studyId or studyLocusId.

    Returns:
        DataFrame: DataFrame with resourceId and count.

    Examples:
        >>> data = [("GCST000001",), ("GCST000002",), ("FINNGEN_R12-0001",)]
        >>> schema = "studyId STRING"
        >>> df = spark.createDataFrame(data, schema)
        >>> describe_resource(df).show()
        +------------+-----+
        |  resourceId|count|
        +------------+-----+
        |GWAS Catalog|    2|
        |     FinnGen|    1|
        +------------+-----+
        <BLANKLINE>

    """
    return classify_by_resource(df).groupBy("resourceId").count()


def classify_by_resource(df: DataFrame) -> DataFrame:
    """Classify the DataFrame by resource.

    Args:
        df (DataFrame): DataFrame containing studyId or studyLocusId.

    Returns:
        DataFrame: DataFrame with resourceId and count.

    """
    expr = f.when(f.lit(False), f.lit(None).cast("string"))
    for x in [ResourceType.GWAS_CATALOG, ResourceType.EQLT_CATALOGUE, ResourceType.UKB_PPP_EUR, ResourceType.FINNGEN]:
        expr = expr.when(
            x.value.filter_expr(),
            f.lit(x.value.name),
        )
    expr = expr.alias("resourceId")
    return df.withColumn("resourceId", expr)

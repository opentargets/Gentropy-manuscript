"""Manuscript methods for the package."""

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f


def group_statistics(df: DataFrame, group_column: Column) -> DataFrame:
    """Calculate group statistics.

    the statistics calculated are:
    * count of each group
    * percentage of each group relative to the total count
    """
    total = df.count()
    grouped_df = (
        df.groupBy(group_column)
        .agg(
            f.count("*").alias("count"),
            f.format_number((f.count("*") / total * 100.0).alias("percentage"), 2).alias("%"),
        )
        .orderBy(f.desc("count"))
    )

    return grouped_df

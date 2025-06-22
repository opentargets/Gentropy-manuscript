"""Manuscript methods for the package."""

from plotnine import (
    aes,
    element_blank,
    element_line,
    element_rect,
    element_text,
    geom_col,
    ggplot,
    labs,
    scale_y_continuous,
    theme,
    scale_fill_brewer,
    geom_histogram,
    coord_cartesian,
    geom_vline,
    annotate,
)
from plotnine import (
    ggplot,
    aes,
    geom_boxplot,
    theme,
    element_text,
    element_line,
    element_rect,
    element_blank,
    labs,
    scale_fill_brewer,
)
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


def aggregated_statistics(df: DataFrame, group_columns: list[str], agg_column: str) -> DataFrame:
    """Calculate aggregated statistics for multiple group columns."""
    total = df.count()
    mean_expr = f.mean(agg_column).alias("mean")
    stddev_expr = f.stddev(agg_column).alias("stddev")
    max_expr = f.max(agg_column).alias("max")
    min_expr = f.min(agg_column).alias("min")

    grouped_df = (
        df.groupBy(group_columns)
        .agg(
            f.count(agg_column).alias("count"),
            mean_expr,
            stddev_expr,
            max_expr,
            min_expr,
        )
        .orderBy(f.desc("count"))
    )

    return grouped_df


def format_scientific_notation(breaks):
    """format_scientific_notation."""
    new_breaks = ["0"]
    other_breaks = [f"{x:.0e}" for x in breaks if x != 0]  # Format as scientific notation without decimal
    new_breaks.extend(other_breaks)
    return new_breaks


def plot_group_statistics(group_stats: DataFrame, x: str, y: str, fill: str, title: str) -> ggplot:
    """Plot grouped statistics."""
    REM = 10
    p = (
        ggplot(data=group_stats.toPandas(), mapping=aes(x=x, y=y, fill=fill))
        + geom_col(stat="identity")
        + labs(x=title)
        + scale_fill_brewer(type="seq", name="% of total")
        + scale_y_continuous(labels=format_scientific_notation)
        + theme(
            figure_size=(5.35, 4.5),  # ~85mm wide
            axis_title=element_text(size=REM * 1, family="sans-serif"),
            axis_text=element_text(size=REM * 0.8, family="sans-serif"),
            axis_text_x=element_text(rotation=45, hjust=1),
            axis_ticks=element_line(color="black"),
            axis_line=element_line(color="black"),
            panel_background=element_rect(fill="white"),
            panel_border=element_rect(color="black", fill=None),
            panel_grid=element_blank(),
            plot_margin=0.1,
        )
    )
    return p


def plot_aggregated_data(df: DataFrame, x: str, y: str, xtitle: str, ytitle: str) -> ggplot:
    """Plot boxplot with aggregated data."""
    data = df.toPandas()
    REM = 10
    plot = (
        ggplot(data)
        + geom_boxplot(aes(x=x, y=y))
        + theme(
            figure_size=(5.35, 4.5),  # ~85mm wide
            axis_title=element_text(size=REM * 1, family="sans-serif"),
            axis_text=element_text(size=REM * 0.8, family="sans-serif"),
            axis_text_x=element_text(rotation=45, hjust=1),
            axis_ticks=element_line(color="black"),
            axis_line=element_line(color="black"),
            panel_background=element_rect(fill="white"),
            panel_border=element_rect(color="black", fill=None),
            panel_grid=element_blank(),
            plot_margin=0.1,
        )
        + labs(x=xtitle, y=ytitle)
        + scale_fill_brewer(type="seq", name="% of total")
    )
    return plot


def plot_distribution(df: DataFrame, factor: str, xtitle: str) -> ggplot:
    """Plot the distribution of credible set sizes."""
    # Convert to Pandas DataFrame for plotting
    dataset = df.toPandas()

    # Plotting parameters
    REM = 10
    mean_val = dataset[factor].mean()

    p = (
        ggplot(data=dataset, mapping=aes(x=factor))
        + geom_histogram(
            bins=50,
            color="grey",
            fill="lightgray",
        )
        + coord_cartesian(ylim=(0, 400))  # limit y-axis to hide the spike
        + labs(x=xtitle)
        + geom_vline(xintercept=mean_val, color="#1f77b4", linetype="dashed", size=0.5, show_legend=True)
        + theme(
            # figure_size=(REM, REM*0.75),
            figure_size=(5.35, 4.5),  # ~85mm wide
            axis_title=element_text(size=REM * 1, family="sans-serif"),
            axis_text=element_text(size=REM * 0.8, family="sans-serif"),
            axis_ticks=element_line(color="black"),
            axis_line=element_line(color="black"),
            panel_background=element_rect(fill="white"),
            panel_border=element_rect(color="black", fill=None),
            panel_grid=element_blank(),
            # plot_margin=0.25,
            plot_margin=0.1,
        )
        + annotate(
            "text",
            x=5000,
            y=50,
            label=f"Mean = {mean_val:.2f}",
            ha="left",
            size=12,
            color="#1f77b4",
            fontweight="bold",
        )
    )
    return p

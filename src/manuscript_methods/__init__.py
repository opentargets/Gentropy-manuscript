"""Manuscript methods for the package."""

import plotnine as p9
from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as f


def group_statistics(df: DataFrame, group_column: Column | list[Column] | str | list[str]) -> DataFrame:
    """Calculate group statistics.

    the statistics calculated are:
    * count of each group
    * percentage of each group relative to the total count
    """
    if isinstance(group_column, str):
        group_column = [f.col(group_column)]
    elif isinstance(group_column, Column):
        group_column = [group_column]
    total = df.count()
    grouped_df = (
        df.groupBy(group_column)
        .agg(f.count("*").alias("count"), (f.count("*") / total * 100.0).alias("percentage"))
        .select(*group_column, f.col("count"), f.format_number(f.col("percentage"), 2).alias("%"), f.col("percentage"))
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


class OpenTargetsTheme:
    """Basic theme for plotnine plots."""

    import plotnine as p9

    REM = 10
    theme = p9.theme(
        figure_size=(5.35, 4.5),
        axis_title=p9.element_text(size=REM * 1, family="sans-serif"),
        axis_text=p9.element_text(size=REM * 0.8, family="sans-serif", rotation=45, hjust=1),
        axis_text_x=p9.element_text(hjust=1),
        axis_text_y=p9.element_text(hjust=1),
        axis_ticks=p9.element_line(color="black"),
        axis_line=p9.element_line(color="black"),
        panel_background=p9.element_rect(fill="white"),
        panel_border=p9.element_rect(color="black", fill=None),
        panel_grid=p9.element_blank(),
    )

    @staticmethod
    def categorical_theme(REM=10):
        return p9.theme(
            axis_title=p9.element_text(size=REM, family="sans-serif"),
            axis_ticks_direction="out",
            axis_ticks=p9.element_line(color="black"),
            axis_ticks_length_major=8,
            axis_ticks_length_minor=4,
            axis_line=p9.element_line(color="black"),
            panel_grid_major=p9.element_blank(),
            panel_grid_minor=p9.element_blank(),
            panel_background=p9.element_rect(fill="white"),
            axis_text_x=p9.element_text(size=REM * 1.5, family="sans-serif", rotation=45, va="top", ha="right"),
            axis_text_y=p9.element_text(size=REM * 2.0, family="sans-serif"),
            axis_title_y=p9.element_text(size=REM * 2.0, family="sans-serif"),
            panel_border=p9.element_rect(color="black", fill=None),
            figure_size=(REM / 2, REM / 1.5),
            legend_position=(0.4, 0.8),
            legend_title=p9.element_text(size=REM),
            legend_text=p9.element_text(size=REM),
            legend_direction="vertical",
            legend_box_background=p9.element_rect(color="black", size=1),
        )

    categorical_light_colors = [
        "#fa4d56",
        "#4589ff",
        "#08bdba",
        "#d4bbff",
        "#007d79",
        "#fff1f1",
        "#d12771",
        "#bae6ff",
        "#8a3ffc",
        "#ff7eb6",
        "#6fdc8c",
        "#d2a106",
        "#ba4e00",
        "#33b1ff",
    ]
    categorical_dark_colors = [
        "#fa4d56",
        "#002d9c",
        "#009d9a",
        "#a56eff",
        "#005d5d",
        "#570408",
        "#ee538b",
        "#012749",
        "#6929c4",
        "#9f1853",
        "#198038",
        "#b28600",
        "#8a3800",
        "#1192e8",
    ]

    association_colors = [
        "#DBEAF6",
        "#BFDAEE",
        "#A5CAE6",
        "#8ABADE",
        "#6EA9D7",
        "#4F97CF",
        "#3583C0",
        "#2C6EA0",
        "#245780",
    ]

    prioritization_colors = [
        "#A01813",
        "#BC3A19",
        "#D65A1F",
        "#E08145",
        "#E3A772",
        "#E6CA9C",
        "#ECEADA",
        "#C5D2C1",
        "#9EBAA8",
        "#78A290",
        "#528B78",
        "#2F735F",
        "#2E5943",
    ]


def plot_group_statistics(group_stats: DataFrame, x: str, y: str, fill: str, title: str) -> p9.ggplot:
    """Plot grouped statistics."""
    p = (
        p9.ggplot(data=group_stats.toPandas(), mapping=p9.aes(x=x, y=y, fill=fill))
        + p9.geom_col(stat="identity")
        + p9.labs(x=title)
        + p9.scale_fill_manual(values=OpenTargetsTheme.categorical_dark_colors)
        + p9.scale_y_continuous(labels=format_scientific_notation)
        + OpenTargetsTheme.categorical_theme(REM=5)
    )
    return p


def plot_aggregated_data(df: DataFrame, x: str, y: str, xtitle: str, ytitle: str) -> p9.ggplot:
    """Plot boxplot with aggregated data."""
    data = df.toPandas()
    REM = 10
    plot = (
        p9.ggplot(data)
        + p9.geom_boxplot(p9.aes(x=x, y=y))
        + OpenTargetsTheme.theme
        + p9.labs(x=xtitle, y=ytitle)
        + p9.scale_fill_brewer(type="seq", name="% of total")
    )
    return plot


def plot_distribution(df: DataFrame, factor: str, xtitle: str) -> p9.ggplot:
    """Plot the distribution of credible set sizes."""
    # Convert to Pandas DataFrame for plotting
    dataset = df.toPandas()

    # Plotting parameters
    REM = 10
    mean_val = dataset[factor].mean()

    p = (
        p9.ggplot(data=dataset, mapping=p9.aes(x=factor))
        + p9.geom_histogram(
            bins=50,
            color="grey",
            fill="lightgray",
        )
        + p9.labs(x=xtitle)
        + p9.geom_vline(xintercept=mean_val, color="#1f77b4", linetype="dashed", size=0.5, show_legend=True)
        + OpenTargetsTheme.theme
        + p9.annotate(
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


def break_string(string: str, nmin: int = 15) -> str:
    """Break a string into multiple lines if the length exceeds nmin characters.

    This function is useful for plots where the group names are too long and
    we need to break them into multiple lines for better readability.
    """
    new_string = ""
    current_rows = 1
    for idx, i in enumerate(string):
        if i == " ":
            if idx - (nmin * current_rows) > 0:
                new_string += "\n"
                current_rows += 1
            else:
                new_string += i
        else:
            new_string += i
    return new_string


def calculate_protein_altering_proportion(vep_score: Column, threshold: float = 0.66) -> Column:
    """Calculate the proportion of protein-altering variants in each bucket."""
    w = Window.partitionBy("bucket", "studyType").orderBy("bucket", "studyType")
    n_protein_altering = f.count(f.when(vep_score >= threshold, 1)).over(w).alias("nAlteringInBucket")
    n_non_protein_altering = f.count(f.when(vep_score < threshold, 1)).over(w).alias("nNonAlteringInBucket")
    total = (n_protein_altering + n_non_protein_altering).alias("totalInBucket")
    proportion = (n_protein_altering / total).alias("alteringProportionInBucket")
    stderr = f.sqrt((proportion * (1 - proportion)) / (total)).alias("stdErr")
    return f.struct(n_protein_altering, n_non_protein_altering, proportion, stderr, total)

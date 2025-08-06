import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    from gentropy import Session
    from pyspark.sql import SparkSession, DataFrame, Column
    from pyspark.sql import functions as f
    from pyspark.sql import types as t
    from manuscript_methods import group_statistics
    import altair as alt
    import pandas as pd
    return Session, alt, group_statistics, pd


@app.cell
def _(Session):
    session = Session()
    spark = session.spark
    return (spark,)


@app.cell
def _():
    dataset_path = "data/25.06/lead_variant_effect/"
    return (dataset_path,)


@app.cell
def _(dataset_path, spark):
    lve = spark.read.parquet(dataset_path)
    return (lve,)


@app.cell
def _(group_statistics, lve):

    df1 = lve.select("studyStatistics.studyType", "majorLdPopulation.ldPopulation")
    gs = group_statistics(df1, group_column=["studyType", "ldPopulation"])



    return (gs,)


@app.cell
def _(gs):
    gs_ = gs.toPandas()
    return (gs_,)


@app.cell
def _(alt, gs_):
    # Altair barplot
    alt.Chart(gs_).mark_bar().encode(x="ldPopulation", y="count", color="studyType")
    return


@app.cell
def _(lve):
    df2 = lve.select(
        "majorLDPopulation.ldPopulation",
        "studyStatistics.studyType",
        "majorLdPopulationAf.alleleFrequency",
    )
    af_df = df2.toPandas()
    k = 1.5
    group_by_column = "ldPopulation"
    value_column = "alleleFrequency"

    return (af_df,)


@app.cell
def _(alt, pd):
    def plot_group_boxplot(df: pd.DataFrame, group_by_column: str, value_column: str, k: float = 1.5):

        agg_stats = df.groupby(group_by_column)[value_column].describe()
        agg_stats["iqr"] = agg_stats["75%"] - agg_stats["25%"]
        agg_stats["min_"] = agg_stats["25%"] - k * agg_stats["iqr"]
        agg_stats["max_"] = agg_stats["75%"] + k * agg_stats["iqr"]
        data_points = df[[value_column, group_by_column]].merge(
            agg_stats.reset_index()[[group_by_column, "min_", "max_"]]
        )
        # Lowest data point which is still above or equal to min_
        # This will be the lower end of the whisker
        agg_stats["lower"] = (
            data_points[data_points[value_column] >= data_points["min_"]]
            .groupby(group_by_column)[value_column]
            .min()
        )
        # Highest data point which is still below or equal to max_
        # This will be the upper end of the whisker
        agg_stats["upper"] = (
            data_points[data_points[value_column] <= data_points["max_"]]
            .groupby(group_by_column)[value_column]
            .max()
        )
        # Store all outliers as a list
        agg_stats["outliers"] = (
            data_points[
                (data_points[value_column] < data_points["min_"])
                | (data_points[value_column] > data_points["max_"])
            ]
            .groupby(group_by_column)[value_column]
            .apply(list)
        )
        agg_stats = agg_stats.reset_index()
        print(agg_stats)

        # Show whole dataframe
        base = alt.Chart(agg_stats).encode(
            y=f"{group_by_column}:N"
        )

        rules = base.mark_rule(color="white").encode(
            x=alt.X("lower").title(value_column),
            x2="upper",
        )

        bars = base.mark_bar(size=14).encode(
            x="25%",
            x2="75%",
            color=alt.Color(group_by_column).legend(None),
        )

        ticks = base.mark_tick(color="white", size=14).encode(
            x="50%"
        )


        plot = rules + ticks  + bars
        return plot
    return (plot_group_boxplot,)


@app.cell
def _(af_df, plot_group_boxplot):
    plot_group_boxplot(af_df, "ldPopulation", "alleleFrequency")
    return


@app.cell
def _(af_df, plot_group_boxplot):
    plot_group_boxplot(af_df, "studyType", "alleleFrequency")
    return


@app.cell
def _():
    import seaborn as sns
    from matplotlib import pyplot as plt

    _new_black = '#373737'
    sns.set_theme(style='ticks', font_scale=0.75, rc={
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'svg.fonttype': 'none',
        'text.usetex': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'axes.labelpad': 2,
        'axes.linewidth': 0.5,
        'axes.titlepad': 4,
        'lines.linewidth': 0.5,
        'legend.fontsize': 9,
        'legend.title_fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.major.size': 2,
        'xtick.major.pad': 1,
        'xtick.major.width': 0.5,
        'ytick.major.size': 2,
        'ytick.major.pad': 1,
        'ytick.major.width': 0.5,
        'xtick.minor.size': 2,
        'xtick.minor.pad': 1,
        'xtick.minor.width': 0.5,
        'ytick.minor.size': 2,
        'ytick.minor.pad': 1,
        'ytick.minor.width': 0.5,

        # Avoid black unless necessary
        'text.color': _new_black,
        'patch.edgecolor': _new_black,
        'patch.force_edgecolor': False, # Seaborn turns on edgecolors for histograms by default and I don't like it
        'hatch.color': _new_black,
        'axes.edgecolor': _new_black,
        # 'axes.titlecolor': _new_black # should fallback to text.color
        'axes.labelcolor': _new_black,
        'xtick.color': _new_black,
        'ytick.color': _new_black

        # Default colormap - personal preference
        # 'image.cmap': 'inferno'
    })
    return


@app.cell
def _(g, pd):
    def plot_group_boxplot_pyplot(df: pd.DataFrame, x:str, y:str, col: str):


        return g
    return (plot_group_boxplot_pyplot,)


@app.cell
def _(af_df, plot_group_boxplot_pyplot):
    p = plot_group_boxplot_pyplot(af_df[af_df["studyType"] == 'gwas'], x="ldPopulation", y="alleleFrequency", col="studyType")
    p

    return


app._unparsable_cell(
    r"""
    def plot_group_boxplot_bokeh(df: pd.DataFrame, x: str, y: str, hue: str):

    """,
    name="_"
)


app._unparsable_cell(
    r"""
    desc = af_df[\"alleleFrequency\"].describe()
    mean_ = desc.loc[\"mean\"]
    count_ desc.loc[\"count\"]
    std_ = desc.loc[\"std\"]
    min_ = desc.loc[\"min\"]
    q25 = desc.loc[\"25%\"]
    q50 = desc.loc[\"50%\"]
    q75 = desc.loc[\"70%\"]
    """,
    name="_"
)


@app.cell
def _(af_df):
    group_col = "ldPopulation"
    val_col = "alleleFrequency"
    grouped = af_df.groupby(group_col)
    statistics = grouped[val_col].describe()
    statistics



    return (statistics,)


@app.cell
def _(statistics):
    statistics.reset_index().to_dict(orient="list")
    return


@app.cell
def _():
    from bokeh.models import ColumnDataSource, Whisker
    from bokeh.plotting import figure, show
    from bokeh.transform import factor_cmap, jitter

    # classes = statistics.index.values
    # p2 = figure(
    #     height=400,
    #     x_range=classes,
    #     background_fill_color="#efefef",
    #     title="Allele Frequency across ld ancestries"
    # )
    # p2.xgrid.grid_line_color = None

    # source = ColumnDataSource(data=statistics.reset_index().to_dict(orient="list"))
    # print(source.__dict__)
    # error = Whisker(base="base", upper="75%", lower="25%", source=source)
    # error.upper_head.size=20
    # error.lower_head.size=20
    # p2.add_layout(error)
    # p2.b

    # show(p2)

    return


@app.cell
def _():
    return


@app.cell
def _():
    from bokeh.sampledata.autompg2 import autompg2 as df
    df
    return


if __name__ == "__main__":
    app.run()

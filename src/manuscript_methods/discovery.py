"""Ancestry classification and cumulative gene-discovery curves.

Used by Results section 1 (Figure 1c), Extended Data Figures 3 and 10, and
Supplementary Results 12.
"""

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as f

ANCESTRY_THRESHOLD = 0.9  # relative sample size a single ancestry must reach to be predominant
MAF_COMMON = 0.01
MAX_YEAR = 2024  # 2025 is a partial year in the release
ANCESTRY_ORDER = ["EUR", "non-EUR", "mixed"]
ANCESTRY_COLORS = {"EUR": "#4472C4", "non-EUR": "#70AD47", "mixed": "#ED7D31", "rare": "#FFC000"}
YEARS = list(range(2006, MAX_YEAR + 1))


def classify_ancestry(df: DataFrame, threshold: float = ANCESTRY_THRESHOLD) -> DataFrame:
    """Label each study EUR, non-EUR or mixed from its LD population structure.

    EUR: one ancestry reaches `threshold` and it is non-Finnish European. non-EUR: one
    ancestry reaches `threshold` and it is another. mixed: no single ancestry reaches it.
    A single listed population with a null relative sample size is the whole sample, so it
    is imputed to 1.0. Studies whose predominant ancestry cannot be determined default to
    EUR, which can only shrink the non-EUR and mixed contributions.
    """
    pops = f.col("ldPopulationStructure")
    top = f.array_max(
        f.transform(pops, lambda x: f.struct(x["relativeSampleSize"].alias("frac"), x["ldPopulation"].alias("pop")))
    )
    nfe_fraction = f.coalesce(
        f.aggregate(
            f.transform(
                f.filter(pops, lambda x: x["ldPopulation"] == f.lit("nfe")),
                lambda x: f.coalesce(x["relativeSampleSize"], f.lit(0.0)),
            ),
            f.lit(0.0),
            lambda acc, x: acc + x,
        ),
        f.lit(0.0),
    )
    return (
        # f.size() returns -1 for a null array under gentropy's session conf, hence the guard.
        df.withColumn("nLdPopulations", f.when(pops.isNull(), f.lit(0)).otherwise(f.size(pops)))
        .withColumn("predominantAncestry", top["pop"])
        .withColumn("rawPredominantFraction", top["frac"])
        .withColumn("nfeFraction", nfe_fraction)
        .withColumn(
            "ldStructureNote",
            f.when(f.col("nLdPopulations") == 0, f.lit("empty_ld"))
            .when(f.col("rawPredominantFraction").isNotNull(), f.lit("defined"))
            .when(f.col("nLdPopulations") == 1, f.lit("single_population_imputed"))
            .otherwise(f.lit("undefined_relative_sample_size")),
        )
        .withColumn(
            "predominantFraction",
            f.when(f.col("ldStructureNote") == "single_population_imputed", f.lit(1.0)).otherwise(
                f.col("rawPredominantFraction")
            ),
        )
        .drop("rawPredominantFraction")
        .withColumn(
            "ancestryClass",
            f.when(f.col("predominantFraction").isNull(), f.lit("EUR"))
            .when(f.col("predominantFraction") < f.lit(threshold), f.lit("mixed"))
            .when(f.col("predominantAncestry") == f.lit("nfe"), f.lit("EUR"))
            .otherwise(f.lit("non-EUR")),
        )
    )


def first_discovery(df: pd.DataFrame, id_cols: list, mask=None) -> pd.DataFrame:
    """First year each entity (gene, or gene-trait pair) appears."""
    sub = df if mask is None else df[mask]
    sub = sub[id_cols + ["year"]].dropna(subset=["year"])
    sub = sub[sub["year"] <= MAX_YEAR].drop_duplicates()
    return sub.groupby(id_cols, as_index=False)["year"].min()


def cumulative_curve(first: pd.DataFrame, years: list = None) -> pd.DataFrame:
    """Per-year new discoveries and their cumulative sum over a fixed year axis."""
    years = YEARS if years is None else years
    per_year = first.groupby("year").size().reindex(years, fill_value=0).rename("count").reset_index()
    per_year = per_year.rename(columns={"index": "year"})
    per_year["cumulative"] = per_year["count"].cumsum()
    return per_year


def explode_pairs(df: pd.DataFrame, trait_col: str = "diseaseIds") -> pd.DataFrame:
    """Explode an array-valued trait column so each row is one gene-trait pair."""
    out = df.explode(trait_col).rename(columns={trait_col: "traitId"})
    return out[out["traitId"].notna()].copy()


def nested_tiers(df: pd.DataFrame, id_cols: list, metric: str) -> pd.DataFrame:
    """Cumulative discovery for nested ancestry tiers, as the stacked figures show them.

    Tiers add ancestry labels in `ANCESTRY_ORDER` over common variants, then a final tier
    adds rare variants of any ancestry. `layer` is the increment over the previous tier,
    which is the height of that stacked bar segment.
    """
    frames = []
    common = df["freqClass"] == "common"
    for k, ancestry in enumerate(ANCESTRY_ORDER, start=1):
        included = ANCESTRY_ORDER[:k]
        mask = common & df["ancestryClass"].isin(included)
        if not mask.any():
            continue
        curve = cumulative_curve(first_discovery(df, id_cols, mask))
        curve["tier"] = " + ".join(included) + " (common)"
        curve["tier_index"] = k
        curve["layer_label"] = f"{ancestry} common"
        frames.append(curve)

    curve = cumulative_curve(first_discovery(df, id_cols))
    curve["tier"] = "all (incl. rare)"
    curve["tier_index"] = len(ANCESTRY_ORDER) + 1
    curve["layer_label"] = "rare"
    frames.append(curve)

    out = pd.concat(frames, ignore_index=True)
    out["metric"] = metric

    wide = out.pivot_table(index="year", columns="tier_index", values="cumulative").sort_index(axis=1)
    layers = wide.diff(axis=1)
    layers[wide.columns[0]] = wide[wide.columns[0]]
    layer_long = layers.stack().rename("layer").reset_index()
    out = out.merge(layer_long, on=["year", "tier_index"], how="left")
    return out[["metric", "tier", "tier_index", "layer_label", "year", "count", "cumulative", "layer"]]


def marginal_strata(df: pd.DataFrame, id_cols: list, metric: str) -> pd.DataFrame:
    """Cumulative discovery within each ancestry by frequency stratum separately."""
    frames = []
    for ancestry in ANCESTRY_ORDER:
        for freq in ["common", "rare"]:
            mask = (df["ancestryClass"] == ancestry) & (df["freqClass"] == freq)
            if not mask.any():
                continue
            curve = cumulative_curve(first_discovery(df, id_cols, mask))
            curve["ancestryClass"] = ancestry
            curve["freqClass"] = freq
            frames.append(curve)
    out = pd.concat(frames, ignore_index=True)
    out["metric"] = metric
    return out[["metric", "ancestryClass", "freqClass", "year", "count", "cumulative"]]


def attribution(df: pd.DataFrame, id_cols: list, metric: str) -> pd.DataFrame:
    """Reachable, exclusive and first-discovery entity counts per ancestry by frequency stratum.

    `reachable`: entities seen at least once in the stratum. `exclusive`: entities seen in
    that stratum only. `first_discovery`: entities whose earliest year falls in the stratum,
    ties broken by `ANCESTRY_ORDER` then common before rare.
    """
    d = df[id_cols + ["ancestryClass", "freqClass", "year"]].dropna(subset=["year"])
    d = d[d["year"] <= MAX_YEAR].copy()
    d["stratum"] = d["ancestryClass"] + " " + d["freqClass"]
    d["entity"] = list(map(tuple, d[id_cols].to_numpy()))

    total = d["entity"].nunique()
    reach = d.groupby("stratum")["entity"].apply(set)
    strata_per_entity = d.groupby("entity")["stratum"].nunique()
    single = set(strata_per_entity[strata_per_entity == 1].index)

    priority = {a: i for i, a in enumerate(ANCESTRY_ORDER)}
    d["ancestry_rank"] = d["ancestryClass"].map(priority)
    d["freq_rank"] = (d["freqClass"] == "rare").astype(int)
    winners = (
        d.sort_values(["year", "ancestry_rank", "freq_rank"])
        .drop_duplicates(subset=["entity"], keep="first")
        .groupby("stratum")
        .size()
    )

    rows = [
        {
            "metric": metric,
            "stratum": stratum,
            "reachable": len(entities),
            "reachable_pct": 100 * len(entities) / total,
            "exclusive": len(entities & single),
            "exclusive_pct": 100 * len(entities & single) / total,
            "first_discovery": int(winners.get(stratum, 0)),
            "first_discovery_pct": 100 * int(winners.get(stratum, 0)) / total,
        }
        for stratum, entities in reach.items()
    ]
    rows.append(
        {
            "metric": metric,
            "stratum": "total",
            "reachable": total,
            "reachable_pct": 100.0,
            "exclusive": len(single),
            "exclusive_pct": 100 * len(single) / total,
            "first_discovery": total,
            "first_discovery_pct": 100.0,
        }
    )
    return pd.DataFrame(rows).sort_values("stratum").reset_index(drop=True)


def rare_share(nested: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rare-variant share of cumulative discovery per year (Extended Data Figure 10)."""
    sub = nested[nested["metric"] == metric]
    wide = sub.pivot_table(index="year", columns="tier_index", values="cumulative").sort_index(axis=1)
    total = wide[wide.columns[-1]]
    rare = total - wide[wide.columns[-2]]
    return pd.DataFrame(
        {"metric": metric, "year": wide.index, "total": total, "rare": rare, "rare_pct": 100 * rare / total}
    ).reset_index(drop=True)


__all__ = [
    "ANCESTRY_COLORS",
    "ANCESTRY_ORDER",
    "MAF_COMMON",
    "MAX_YEAR",
    "YEARS",
    "attribution",
    "classify_ancestry",
    "cumulative_curve",
    "explode_pairs",
    "first_discovery",
    "marginal_strata",
    "nested_tiers",
    "rare_share",
]

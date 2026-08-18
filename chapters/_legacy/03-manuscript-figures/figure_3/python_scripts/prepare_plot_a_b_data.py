"""
Prepare and export the data behind Plot a and Plot b from `clustering_analysis.ipynb`.

Outputs:
- `chapters/03-manuscript-figures/figure_3/data/plot_a.csv`
- `chapters/03-manuscript-figures/figure_3/data/plot_b.csv`

This script intentionally includes only the minimal pipeline required to reproduce the
data used in:
- Plot a: regression coefficients (univariate vs joint model) -> `combined_results`
- Plot b: binned observed vs predicted traits by MAF bin -> `binned_data`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyspark.sql.functions as f
import pyspark.sql.types as t
import statsmodels.api as sm
from gentropy.common.session import Session
from pyspark.sql.window import Window
from scipy.stats import ncx2


def cluster_lead_variants(lead_variants_list: list[dict], colocalisation_list: list[dict]) -> list[dict]:
    """Cluster study loci based on colocalisation edges and shared lead variants."""
    # Lookup for studyLocusId -> info dict
    study_locus_info = {item["studyLocusId"]: item for item in lead_variants_list}

    # variantId -> list[studyLocusId]
    variant_to_loci: dict[str, list[str]] = {}
    for item in lead_variants_list:
        variant_id = item["leadVariantId"]
        variant_to_loci.setdefault(variant_id, []).append(item["studyLocusId"])

    coloc_lookup: dict[str, set[str]] = {}
    for item in colocalisation_list:
        left_id = item["leftStudyLocusId"]
        right_id = item["rightStudyLocusId"]
        coloc_lookup.setdefault(left_id, set()).add(right_id)
        coloc_lookup.setdefault(right_id, set()).add(left_id)

    clusters: list[dict] = []
    used_loci: set[str] = set()

    for lead_variant in lead_variants_list:
        start_locus_id = lead_variant["studyLocusId"]
        if start_locus_id in used_loci:
            continue

        cluster_loci_to_process = {start_locus_id}
        final_cluster_loci: set[str] = set()

        while cluster_loci_to_process:
            locus_id = cluster_loci_to_process.pop()
            if locus_id in used_loci:
                continue

            final_cluster_loci.add(locus_id)
            used_loci.add(locus_id)

            # Colocalised neighbours
            for neighbour in coloc_lookup.get(locus_id, set()):
                if neighbour not in used_loci:
                    cluster_loci_to_process.add(neighbour)

            # Shared lead variant
            info = study_locus_info.get(locus_id)
            if info is not None:
                variant_id = info["leadVariantId"]
                for shared_variant_locus in variant_to_loci.get(variant_id, []):
                    if shared_variant_locus not in used_loci:
                        cluster_loci_to_process.add(shared_variant_locus)

        cluster_details = [
            study_locus_info[locus_id] for locus_id in final_cluster_loci if locus_id in study_locus_info
        ]
        if cluster_details:
            clusters.append(
                {
                    "cluster_id": len(clusters),
                    "lead_study_locus_id": start_locus_id,
                    "colocalised_study_loci": cluster_details,
                    "cluster_size": len(cluster_details),
                }
            )

    return clusters


def build_cluster_pleiotropy(session: Session):
    """Build the Spark DataFrame used downstream for Plot a and Plot b."""
    coloc = session.spark.read.parquet("/Users/polina/Gentropy-manuscript/data/colocalisation_coloc")
    ecaviar = session.spark.read.parquet("/Users/polina/Gentropy-manuscript/data/colocalisation_ecaviar")

    lead_variants = (
        session.spark.read.parquet("/Users/polina/Gentropy-manuscript/data/qualifying_credible_sets")
        .withColumn(
            "pValue",
            f.col("variantStatistics.pValueMantissa") * f.pow(10, f.col("variantStatistics.pValueExponent")),
        )
        .sort("pValue", f.desc("locusStatistics.leadVariantPIP"))
        .persist()
    )

    study_loci_list = [
        {
            "studyLocusId": row["studyLocusId"],
            "leadVariantId": row["variantId"],
            "traitId": row["traitFromSourceMappedIds"],
        }
        for row in lead_variants.select("studyLocusId", "variantId", "traitFromSourceMappedIds").collect()
    ]
    study_locus_ids = [d["studyLocusId"] for d in study_loci_list]

    colocalisation = (
        coloc.unionByName(ecaviar, True)
        .filter((f.col("h4") >= 0.8) | (f.col("clpp") >= 0.01))
        .select("leftStudyLocusId", "rightStudyLocusId")
        .filter(f.col("leftStudyLocusId").isin(study_locus_ids) & f.col("rightStudyLocusId").isin(study_locus_ids))
        .persist()
    )
    colocalisation_list = [
        {
            "leftStudyLocusId": row["leftStudyLocusId"],
            "rightStudyLocusId": row["rightStudyLocusId"],
        }
        for row in colocalisation.collect()
    ]

    clusters = cluster_lead_variants(study_loci_list, colocalisation_list)

    schema = t.StructType(
        [
            t.StructField("studyLocusId", t.StringType(), True),
            t.StructField("leadVariantId", t.StringType(), True),
            t.StructField("traitId", t.ArrayType(t.StringType()), True),
        ]
    )
    cluster_schema = t.StructType(
        [
            t.StructField("cluster_id", t.IntegerType(), True),
            t.StructField("lead_study_locus_id", t.StringType(), True),
            t.StructField("colocalised_study_loci", t.ArrayType(schema), True),
            t.StructField("cluster_size", t.IntegerType(), True),
        ]
    )

    clusters_df = session.spark.createDataFrame(clusters, schema=cluster_schema).select(
        "cluster_id", "lead_study_locus_id", "cluster_size", "colocalised_study_loci"
    )

    clusters_with_trait_count = clusters_df.withColumn(
        "uniqueTraitsInCluster",
        f.size(f.array_distinct(f.flatten(f.col("colocalised_study_loci.traitId")))),
    )
    exploded_loci = clusters_with_trait_count.select("cluster_id", f.explode("colocalised_study_loci").alias("locus"))
    variant_traits_in_cluster = exploded_loci.groupBy("cluster_id", "locus.leadVariantId").agg(
        f.size(f.array_distinct(f.flatten(f.collect_list("locus.traitId")))).alias("uniqueTraitCountForVariant")
    )
    window_spec = Window.partitionBy("cluster_id").orderBy(f.desc("uniqueTraitCountForVariant"), f.col("leadVariantId"))
    top_variant_per_cluster = (
        variant_traits_in_cluster.withColumn("row_num", f.row_number().over(window_spec))
        .filter(f.col("row_num") == 1)
        .select(
            f.col("cluster_id"),
            f.col("leadVariantId").alias("clusterVariantId"),
        )
    )
    result_df = (
        clusters_with_trait_count.join(top_variant_per_cluster, "cluster_id")
        .select(
            "cluster_id",
            "cluster_size",
            "colocalised_study_loci",
            "uniqueTraitsInCluster",
            "clusterVariantId",
        )
        .sort("cluster_id")
        .persist()
    )

    variant_pleiotropy_clustered = session.spark.read.parquet(
        "gs://genetics-portal-dev-analysis/dc16/output/gentropy_paper/variant_pleiotropy"
    ).join(result_df, f.col("variantId") == f.col("clusterVariantId"), "inner")

    @f.udf(t.DoubleType())
    def chi2_sf_udf(ncp_val):
        """Survival function (1-CDF) for a non-central chi-squared distribution."""
        if ncp_val is None:
            return None
        return float(ncx2.sf(x=32.84125, df=1, nc=ncp_val))

    cluster_pleiotropy = (
        variant_pleiotropy_clustered.withColumn(
            "ncp",
            (f.pow(f.col("maxAbsBeta"), 2) * f.col("maxEffectiveSampleSize") * f.col("maxVarG")) / 11,
        )
        .withColumn("predictedPower", chi2_sf_udf(f.col("ncp")))
        .persist()
    )

    return cluster_pleiotropy


def build_plot_a_and_b(cluster_pleiotropy) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (plot_a_df, plot_b_df)."""
    df = cluster_pleiotropy.toPandas()[
        [
            "uniqueTraitsInCluster",
            "maxAbsBeta",
            "maxMAF",
            "maxEffectiveSampleSize",
            "gerpNormalised",
            "vepScore",
            "predictedPower",
            "cluster_size",
        ]
    ].copy()

    df["gerpNormalised"] = df["gerpNormalised"].fillna(df["gerpNormalised"].mean())
    df["vepBinary"] = (df["vepScore"] >= 0.66).astype(int)

    for col in [
        "maxAbsBeta",
        "maxMAF",
        "gerpNormalised",
        "vepBinary",
        "maxEffectiveSampleSize",
        "predictedPower",
    ]:
        df[f"{col}Normalised"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    covariates_univariate = [
        "maxAbsBetaNormalised",
        "maxMAFNormalised",
        "maxEffectiveSampleSizeNormalised",
        "gerpNormalisedNormalised",
        "vepBinaryNormalised",
        "predictedPowerNormalised",
    ]
    covariates_multi = list(covariates_univariate)

    # Plot a data (combined_results)
    univariate_results = []
    for covariate in covariates_univariate:
        x_uni = sm.add_constant(df[[covariate]].copy())
        y_uni = df["uniqueTraitsInCluster"]
        model_uni = sm.NegativeBinomial(y_uni, x_uni).fit(disp=False, maxiter=1000)
        univariate_results.append(
            {
                "covariate": covariate,
                "coefficient": model_uni.params[covariate],
                "std_error": model_uni.bse[covariate],
                "p_value": model_uni.pvalues[covariate],
                "ci_lower": model_uni.conf_int().loc[covariate, 0],
                "ci_upper": model_uni.conf_int().loc[covariate, 1],
            }
        )
    results_df = pd.DataFrame(univariate_results)

    x_multi = sm.add_constant(df[covariates_multi].copy())
    y_multi = df["uniqueTraitsInCluster"]
    model_multi = sm.NegativeBinomial(y_multi, x_multi).fit(disp=False, maxiter=1000)

    multi_results = []
    for covariate in covariates_univariate:
        if covariate in covariates_multi:
            multi_results.append(
                {
                    "covariate": covariate,
                    "coefficient": model_multi.params[covariate],
                    "std_error": model_multi.bse[covariate],
                    "p_value": model_multi.pvalues[covariate],
                    "ci_lower": model_multi.conf_int().loc[covariate, 0],
                    "ci_upper": model_multi.conf_int().loc[covariate, 1],
                }
            )
        else:
            multi_results.append(
                {
                    "covariate": covariate,
                    "coefficient": np.nan,
                    "std_error": np.nan,
                    "p_value": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                }
            )
    multi_df = pd.DataFrame(multi_results)

    results_df["model_type"] = "Univariate"
    multi_df["model_type"] = "Multi"
    combined_results = pd.concat([results_df, multi_df], ignore_index=True)

    y_mapping = {cov: i for i, cov in enumerate(covariates_univariate)}
    combined_results["y_numerical"] = combined_results["covariate"].map(y_mapping)
    offset_val = 0.1
    combined_results["y_plot"] = combined_results["y_numerical"] + np.where(
        combined_results["model_type"] == "Univariate", -offset_val, offset_val
    )

    covariate_label_map = {
        "maxAbsBetaNormalised": "Absolute beta",
        "maxMAFNormalised": "MAF",
        "maxEffectiveSampleSizeNormalised": "Sample size",
        "gerpNormalisedNormalised": "GERP",
        "vepBinaryNormalised": "PAV",
        "predictedPowerNormalised": "Predicted power",
    }
    combined_results["covariate_label"] = combined_results["covariate"].map(covariate_label_map)

    plot_a_df = combined_results[
        [
            "covariate",
            "covariate_label",
            "model_type",
            "coefficient",
            "std_error",
            "p_value",
            "ci_lower",
            "ci_upper",
            "y_numerical",
            "y_plot",
        ]
    ].copy()

    # Plot b data (binned_data)
    predicted_values = model_multi.predict(x_multi)

    covariates_multi_new = [
        "maxAbsBetaNormalised",
        "maxMAFNormalised",
        "maxEffectiveSampleSizeNormalised",
        "gerpNormalisedNormalised",
        "vepBinaryNormalised",
    ]
    x_multi_new = sm.add_constant(df[covariates_multi_new].copy())
    model_multi_new = sm.NegativeBinomial(y_multi, x_multi_new).fit(disp=False, maxiter=1000)
    predicted_multi_new = model_multi_new.predict(x_multi_new)

    df["predicted_traits_full_model"] = predicted_values
    df["predicted_traits_no_power"] = predicted_multi_new

    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    labels = ["0-0.01", "0.01-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5"]
    df["maxMAF_bin"] = pd.cut(df["maxMAF"], bins=bins, labels=labels, right=False)

    binned_data = (
        df.groupby("maxMAF_bin")
        .agg(
            observed_mean=("uniqueTraitsInCluster", "mean"),
            observed_sem=("uniqueTraitsInCluster", "sem"),
            predicted_full_mean=("predicted_traits_full_model", "mean"),
            predicted_full_sem=("predicted_traits_full_model", "sem"),
            predicted_no_power_mean=("predicted_traits_no_power", "mean"),
            predicted_no_power_sem=("predicted_traits_no_power", "sem"),
        )
        .reset_index()
    )

    # Keep bin labels as strings in CSV (friendlier for R/ggplot2)
    plot_b_df = binned_data.copy()
    plot_b_df["maxMAF_bin"] = plot_b_df["maxMAF_bin"].astype(str)

    return plot_a_df, plot_b_df


def main() -> None:
    session = Session(
        extended_spark_conf={
            "spark.executor.memory": "10g",
            "spark.driver.memory": "10g",
            "spark.driver.maxResultSize": "0",
        }
    )

    cluster_pleiotropy = build_cluster_pleiotropy(session)
    plot_a_df, plot_b_df = build_plot_a_and_b(cluster_pleiotropy)

    figure_3_dir = Path(__file__).resolve().parents[1]
    out_dir = figure_3_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_a_path = out_dir / "plot_b.csv"  # ex-plot_a.csv
    plot_b_path = out_dir / "plot_a.csv"  # ex-plot_b.csv

    plot_a_df.to_csv(plot_a_path, index=False)
    plot_b_df.to_csv(plot_b_path, index=False)

    print(f"Wrote: {plot_a_path}")
    print(f"Wrote: {plot_b_path}")


if __name__ == "__main__":
    main()

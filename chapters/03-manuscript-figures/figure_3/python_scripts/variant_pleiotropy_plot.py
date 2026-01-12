import pyspark.sql.functions as f
import numpy as np
import matplotlib.pyplot as plt

from gentropy.common.session import Session


def main() -> None:
    # ============================
    # Dependencies / PySpark inputs
    # ============================
    session = Session(extended_spark_conf={"spark.driver.memory": "13g"})

    # Core study- and variant-level inputs
    studies = session.spark.read.parquet("/Users/polina/Gentropy-manuscript/data/gwas_therapeutic_areas").filter(
        ~f.col("measurement") & f.col("binaryLessCases")
    )

    qualified_cs = session.spark.read.parquet("/Users/polina/Gentropy-manuscript/data/qualifying_credible_sets")

    # Disease / therapeutic area lookups
    disease_df = session.spark.read.parquet("/Users/polina/Gentropy-manuscript/data/disease.parquet").select(
        f.col("id"), f.col("name")
    )

    # ============================
    # Analysis / data preparation
    # ============================
    # Map diseaseId -> diseaseNames per study
    disease_id_to_name = (
        studies.withColumn("exploded_disease_id", f.explode("diseaseIds"))
        .join(
            disease_df.alias("d"),
            f.col("exploded_disease_id") == f.col("d.id"),
            "left",
        )
        .groupBy("studyId")
        .agg(f.collect_list("name").alias("diseaseNames"))
    )

    # Map therapeuticAreaId -> therapeuticAreaNames per study
    ta_id_to_name = (
        studies.withColumn("exploded_ta_id", f.explode("mappedTherapeuticAreas"))
        .join(
            disease_df.alias("t"),
            f.col("exploded_ta_id") == f.col("t.id"),
            "left",
        )
        .withColumn(
            "ta_name",
            f.when(f.col("exploded_ta_id") == "other", f.lit("other")).otherwise(f.col("t.name")),
        )
        .groupBy("studyId")
        .agg(f.collect_list("ta_name").alias("therapeuticAreaNames"))
    )

    # Enrich studies with disease and therapeutic area names
    studies_with_names = studies.join(disease_id_to_name, "studyId", "left").join(ta_id_to_name, "studyId", "left")

    # Subset to the variant of interest and extract fields for plotting
    df_for_plot = (
        # qualified_cs.filter(f.col("variantId") == "19_44908684_T_C")
        qualified_cs.filter(f.col("variantId") == "19_44908822_C_T")
        .filter(f.col("originalBeta").isNotNull())
        .join(
            studies_with_names.select(
                "studyId",
                "diseaseIds",
                "diseaseNames",
                "mappedTherapeuticAreas",
                "therapeuticAreaNames",
            ),
            "studyId",
            "left",
        )
        .select(
            "studyId",
            "diseaseIds",
            "diseaseNames",
            "mappedTherapeuticAreas",
            "therapeuticAreaNames",
            "rescaledStatistics.estimatedBeta",
            "variantStatistics.pValueMantissa",
            "variantStatistics.pValueExponent",
        )
        .toPandas()
    )

    # Compute -log10(p-value)
    df_for_plot["neg_log10_p"] = -(np.log10(df_for_plot["pValueMantissa"]) + df_for_plot["pValueExponent"])

    # ============================
    # Plot generation
    # ============================
    # Explode the mapped therapeutic areas so each TA gets its own point
    df_for_plot_exploded = df_for_plot.explode("therapeuticAreaNames")

    # Create a color map for therapeutic areas
    unique_tas = df_for_plot_exploded["therapeuticAreaNames"].unique()
    colors = plt.cm.get_cmap("tab20", len(unique_tas))
    ta_color_map = {ta: colors(i) for i, ta in enumerate(unique_tas)}

    plt.figure(figsize=(14, 8))

    # Plot each therapeutic area with a different color
    for ta, color in ta_color_map.items():
        subset = df_for_plot_exploded[df_for_plot_exploded["therapeuticAreaNames"] == ta]
        plt.scatter(
            subset["estimatedBeta"],
            subset["neg_log10_p"],
            label=ta,
            color=color,
            s=100,
        )

    plt.xlabel("estimatedBeta")
    plt.ylabel("-log10(p-value)")
    plt.title("estimatedBeta vs. -log10(p-value) for variant 19_44908684_T_C")
    plt.axvline(0, color="red", linestyle="--")
    plt.grid(True)
    plt.legend(title="Therapeutic Area", loc="upper left")
    plt.tight_layout()
    plt.show()

    # Save exploded data to CSV as well
    df_for_plot_exploded.to_csv("variant_pleiotropy_data_exploded_2.csv", index=False)
    print("Exploded data saved to variant_pleiotropy_data_exploded.csv")


if __name__ == "__main__":
    main()

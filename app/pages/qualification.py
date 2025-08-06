"""Study and credible set qualification."""

import os

import streamlit as st
from gentropy import Session
from pyspark.sql import functions as f
from pyspark.sql import types as t

from manuscript_methods import group_statistics

st.set_page_config(
    page_title="Systematic ancestry-specific fine-mapping",
)
st.title("Systematic ancestry-specific fine-mapping")

THERAPEUTIC_AREA_HIEARACHY = {
    "EFO_0001444": "measurement",
    "MONDO_0045024": "cancer or benign tumor",
    "OTAR_0000018": "genetic, familial or congenital disease",
    "EFO_0005741": "infectious disease",
    "OTAR_0000009": "injury, poisoning or other complication",
    "OTAR_0000014": "pregnancy or perinatal disease",
    "MONDO_0024458": "disorder of visual system",
    "EFO_0000319": "cardiovascular disease",
    "EFO_0009605": "pancreas disease",
    "EFO_0010282": "gastrointestinal disease",
    "OTAR_0000017": "reproductive system or breast disease",
    "EFO_0010285": "integumentary system disease",
    "EFO_0001379": "endocrine system disease",
    "OTAR_0000010": "respiratory or thoracic disease",
    "EFO_0009690": "urinary system disease",
    "OTAR_0000006": "musculoskeletal or connective tissue disease",
    "MONDO_0021205": "disorder of ear",
    "EFO_0000540": "immune system disease",
    "EFO_0005803": "hematologic disease",
    "EFO_0000618": "nervous system disease",
    "MONDO_0002025": "psychiatric disorder",
    "OTAR_0000020": "nutritional or metabolic disease",
    "EFO_0003765": "sign or symptom",  # Not a therapeutic area - is descendant of phenotype
    # "EFO_0000651": "phenotype",
    # "GO_0008150":  "biological process",
    # "EFO_0002571":  "medical procedure",
    # "EFO_0005932": "animal disease",
}


session = st.session_state.get("session", Session(extended_spark_conf={"spark.driver.memory": "50g"}))

datasets = st.session_state.get("datasets", None)

if datasets and session:
    st.write("## Therapeutic areas breakdown")

    st.markdown("""
    The following table shows the number of studies defined by therapeutic areas.""")

    @f.udf(t.StringType())
    def get_first_matching_therapeutic_area(therapeutic_areas_list):
        if therapeutic_areas_list is None:
            return None
        for ta in THERAPEUTIC_AREA_HIEARACHY:
            if ta in therapeutic_areas_list:
                return ta
        return None

    # These lines create a dictionary of diseaseId to primary therapeutic area
    st.write("### Disease therapeutic areas")
    efo_ta = (
        datasets["disease"]
        .select("id", "ancestors")
        .withColumn("primaryTherapeuticArea", get_first_matching_therapeutic_area(f.col("ancestors")))
        .withColumn(
            "primaryTherapeuticArea",
            f.when(f.col("primaryTherapeuticArea").isNull(), f.lit("other")).otherwise(f.col("primaryTherapeuticArea")),
        )
        .join(datasets["study"].select(f.explode("diseaseIds").alias("efo")), f.col("id") == f.col("efo"), "semi")
    )
    efo_ta_lookup = efo_ta.select("id", "primaryTherapeuticArea").collect()
    efo_ta_dict = {row["id"]: row["primaryTherapeuticArea"] for row in efo_ta_lookup}

    # This udf takes a diseaseIds arrays and creates an array of mapped therapeutic areas
    @f.udf(t.ArrayType(t.StringType()))
    def map_efos_to_therapeutic_areas(efo_ids):
        if efo_ids is None:
            return None
        lookup_dict = efo_ta_dict
        mapped_areas = []
        for efo_id in efo_ids:
            mapped_areas.append(lookup_dict.get(efo_id, None))
            mapped_areas = list(set(area for area in mapped_areas if area is not None))
        return mapped_areas

    gwas = (
        datasets["study"]
        .filter(f.col("studyType") == "gwas")
        .withColumn("mappedTherapeuticAreas", map_efos_to_therapeutic_areas(f.col("diseaseIds")))
        .withColumn("measurement", f.array_contains("mappedTherapeuticAreas", "EFO_0001444"))
        .withColumn(
            "binaryLessCases",
            f.when(f.col("nCases") < f.col("nControls"), True).otherwise(False),
        )
        .withColumns(
            {
                "cancerOrBenignTumor": f.when(f.array_contains("mappedTherapeuticAreas", "MONDO_0045024"), 1).otherwise(
                    0
                ),
                "infectiousDisease": f.when(f.array_contains("mappedTherapeuticAreas", "EFO_0005741"), 1).otherwise(0),
                "pregnancyOrPerinatalDisease": f.when(
                    f.array_contains("mappedTherapeuticAreas", "OTAR_0000014"), 1
                ).otherwise(0),
                "disorderOfVisualSystem": f.when(
                    f.array_contains("mappedTherapeuticAreas", "MONDO_0024458"), 1
                ).otherwise(0),
                "cardiovascularDisease": f.when(f.array_contains("mappedTherapeuticAreas", "EFO_0000319"), 1).otherwise(
                    0
                ),
                "pancreasDisease": f.when(f.array_contains("mappedTherapeuticAreas", "EFO_0009605"), 1).otherwise(0),
                "gastrointestinalDisease": f.when(
                    f.array_contains("mappedTherapeuticAreas", "EFO_0010282"), 1
                ).otherwise(0),
                "reproductiveSystemOrBreastDisease": f.when(
                    f.array_contains("mappedTherapeuticAreas", "OTAR_0000017"), 1
                ).otherwise(0),
                "integumentarySystemDisease": f.when(
                    f.array_contains("mappedTherapeuticAreas", "EFO_0010285"), 1
                ).otherwise(0),
                "endocrineSystemDisease": f.when(
                    f.array_contains("mappedTherapeuticAreas", "EFO_0001379"), 1
                ).otherwise(0),
                "respiratoryOrThoracicDisease": f.when(
                    f.array_contains("mappedTherapeuticAreas", "OTAR_0000010"), 1
                ).otherwise(0),
                "urinarySystemDisease": f.when(f.array_contains("mappedTherapeuticAreas", "EFO_0009690"), 1).otherwise(
                    0
                ),
                "musculoskeletalOrConnectiveTissueDisease": f.when(
                    f.array_contains("mappedTherapeuticAreas", "OTAR_0000006"), 1
                ).otherwise(0),
                "disorderOfEar": f.when(f.array_contains("mappedTherapeuticAreas", "MONDO_0021205"), 1).otherwise(0),
                "immuneSystemDisease": f.when(f.array_contains("mappedTherapeuticAreas", "EFO_0000540"), 1).otherwise(
                    0
                ),
                "hematologicDisease": f.when(f.array_contains("mappedTherapeuticAreas", "EFO_0005803"), 1).otherwise(0),
                "nervousSystemDisease": f.when(f.array_contains("mappedTherapeuticAreas", "EFO_0000618"), 1).otherwise(
                    0
                ),
                "psychiatricDisorder": f.when(f.array_contains("mappedTherapeuticAreas", "MONDO_0002025"), 1).otherwise(
                    0
                ),
                "nutritionalOrMetabolicDisease": f.when(
                    f.array_contains("mappedTherapeuticAreas", "OTAR_0000020"), 1
                ).otherwise(0),
                "geneticFamilialOrCongenitalDisease": f.when(
                    f.array_contains("mappedTherapeuticAreas", "OTAR_0000018"), 1
                ).otherwise(0),
                "injuryPoisoningOrOtherComplication": f.when(
                    f.array_contains("mappedTherapeuticAreas", "OTAR_0000009"), 1
                ).otherwise(0),
                "signOrSymptom": f.when(f.array_contains("mappedTherapeuticAreas", "EFO_0003765"), 1).otherwise(0),
                "other": f.when(f.array_contains("mappedTherapeuticAreas", "other"), 1).otherwise(0),
            }
        )
        .withColumn(
            "totalTherapeuticAreas",
            f.col("cancerOrBenignTumor")
            + f.col("infectiousDisease")
            + f.col("pregnancyOrPerinatalDisease")
            + f.col("disorderOfVisualSystem")
            + f.col("cardiovascularDisease")
            + f.col("pancreasDisease")
            + f.col("gastrointestinalDisease")
            + f.col("reproductiveSystemOrBreastDisease")
            + f.col("integumentarySystemDisease")
            + f.col("endocrineSystemDisease")
            + f.col("respiratoryOrThoracicDisease")
            + f.col("urinarySystemDisease")
            + f.col("musculoskeletalOrConnectiveTissueDisease")
            + f.col("disorderOfEar")
            + f.col("immuneSystemDisease")
            + f.col("hematologicDisease")
            + f.col("nervousSystemDisease")
            + f.col("psychiatricDisorder")
            + f.col("nutritionalOrMetabolicDisease")
            + f.col("geneticFamilialOrCongenitalDisease")
            + f.col("injuryPoisoningOrOtherComplication")
            + f.col("signOrSymptom")
            + f.col("other"),
        )
    )
    st.dataframe(
        group_statistics(gwas.filter(f.col("binaryLessCases")), "measurement")
        .select(f.format_number("count", 2).alias("count"), "measurement", "%")
        .toPandas()
        .set_index("measurement")
    )

    qualifying_studies = (
        gwas.filter(f.col("binaryLessCases"))
        .filter(~f.col("measurement"))
        .filter(f.col("nSamples") > 10_000)
        .filter((f.col("nCases") / f.col("nSamples")) >= 0.005)
    )

    # Removing protein measurements and microbiome descendants
    # EFO_0007882 - microbiome
    # EFO_0004747 - protein measurement

    proteins_and_microbiome = (
        datasets["disease"]
        .select("id", "descendants")
        .filter(f.col("id").isin(["EFO_0007882", "EFO_0004747"]))
        .select(f.explode("descendants"))
    )
    proteins_and_microbiome_ids = [row["col"] for row in proteins_and_microbiome.collect()]
    proteins_and_microbiome_ids.extend(["EFO_0007882", "EFO_0004747"])

    qualifying_measurements = (
        gwas.filter(f.col("measurement"))
        .filter(~f.col("binaryLessCases"))
        .filter(f.size(f.array_intersect(f.col("diseaseIds"), f.lit(proteins_and_microbiome_ids))) == 0)
    )
    st.write("### Qualifying studies")
    st.write(f"Number of qualifying studies: {qualifying_studies.count()}")
    st.write(f"Number of qualifying measurements: {qualifying_measurements.count()}")

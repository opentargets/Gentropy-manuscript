"""Therapeutic area methods."""

from enum import StrEnum

from gentropy.common.spark import string2camelcase
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql import types as t


class TherapeuticAreaHierarchy(StrEnum):
    """Therapeutic areas hierarchy.

    Note: The order of the entries matters, as it defines the priority when multiple therapeutic areas match.
    """

    EFO_0001444 = "measurement"
    MONDO_0045024 = "cancer or benign tumor"
    OTAR_0000018 = "genetic familial or congenital disease"
    EFO_0005741 = "infectious disease"
    OTAR_0000009 = "injury poisoning or other complication"
    OTAR_0000014 = "pregnancy or perinatal disease"
    MONDO_0024458 = "disorder of visual system"
    EFO_0000319 = "cardiovascular disease"
    EFO_0009605 = "pancreas disease"
    EFO_0010282 = "gastrointestinal disease"
    OTAR_0000017 = "reproductive system or breast disease"
    EFO_0010285 = "integumentary system disease"
    EFO_0001379 = "endocrine system disease"
    OTAR_0000010 = "respiratory or thoracic disease"
    EFO_0009690 = "urinary system disease"
    OTAR_0000006 = "musculoskeletal or connective tissue disease"
    MONDO_0021205 = "disorder of ear"
    EFO_0000540 = "immune system disease"
    EFO_0005803 = "hematologic disease"
    EFO_0000618 = "nervous system disease"
    MONDO_0002025 = "psychiatric disorder"
    OTAR_0000020 = "nutritional or metabolic disease"
    EFO_0003765 = "sign or symptom"  # Not a therapeutic area - is descendant of phenotype
    # "EFO_0000651": "phenotype",
    # "GO_0008150":  "biological process",
    # "EFO_0002571":  "medical procedure",
    # "EFO_0005932": "animal disease",


def get_first_matching_therapeutic_area(
    ancestors: Column, therapeutic_area_hierarchy: type[TherapeuticAreaHierarchy] = TherapeuticAreaHierarchy
) -> Column:
    """Find the EFO entry ancestor that matches to the FIRST entry in therapeutic area hierarchy.

    Given the column with ancestor EFO entries, we

    Args:
        ancestors (Column): Column with EFO entry ancestries.
        therapeutic_area_hierarchy (type[TherapeuticArea]): Enum with therapeutic areas

    Returns:
        Column: with therapeutic area that match the entry with the lowest index in hierarchy.

    In case there are multiple matches, return the one with the lowest index in the enum.

    Examples:
    --------
    >>> data = [(["EFO_000001", "EFO_0001444", "MONDO_0045024"],),]
    >>> schema = t.StructType([t.StructField("ancestors", t.ArrayType(t.StringType()), True)])
    >>> ancestor_df = spark.createDataFrame(data, schema)
    >>> ancestor_df.show(truncate=False)
    +----------------------------------------+
    |ancestors                               |
    +----------------------------------------+
    |[EFO_000001, EFO_0001444, MONDO_0045024]|
    +----------------------------------------+
    >>> ancestor_df.withColumn("primaryTherapeuticArea", get_first_matching_therapeutic_area(f.col("ancestors"))).show(truncate=False)
    +----------------------------------------+----------------------+
    |ancestors                               |primaryTherapeuticArea|
    +----------------------------------------+----------------------+
    |[EFO_000001, EFO_0001444, MONDO_0045024]|EFO_0001444           |
    +----------------------------------------+----------------------+

    """
    tas = f.array(*[f.lit(ta.name).alias("name") for ta in therapeutic_area_hierarchy])
    # Overlap tas and therapeutic_areas_array
    # The intersection will contain all of the therapeutic areas that match the ancestors
    # This step drops terms that are not expected to be in the ancestors
    intersection = f.array_intersect(ancestors, tas)
    # Prepare hierarchy index
    hierarchy_index = f.array(
        *[
            f.struct(f.lit(ta.name).alias("name"), f.lit(idx).alias("index"))
            for idx, ta in enumerate(therapeutic_area_hierarchy)
        ]
    )
    # Filter the index by the intersection
    intersection_index = (
        f.filter(hierarchy_index, lambda x: f.array_contains(intersection, x.getField("name")))
        .getItem(0)
        .getField("name")
    )
    return intersection_index


def get_efo_ta_index(
    disease: DataFrame, ta_hierarchy: type[TherapeuticAreaHierarchy] = TherapeuticAreaHierarchy
) -> DataFrame:
    """Get the EFO-Therapeutic Area lookup table.

    This table maps the EFO disease id to it's primary therapeutic area found in the hierarchy based on ancestors.
    In case no therapeutic area is found, the "other" value is assigned.

    Args:
        disease (DataFrame): DataFrame with EFO disease entries with "id" and "ancestors" columns.
        ta_hierarchy (type[TherapeuticAreaHierarchy]): Enum with therapeutic areas.

    Returns:
        DataFrame: with "id" and "primaryTherapeuticArea" columns.

    Examples:
    --------
    >>> data = [("EFO_000001", ["EFO_000001", "EFO_0001444"]), ("EFO_000002", ["EFO_000002"]),]
    >>> schema = t.StructType([t.StructField("id", t.StringType(), True), t.StructField("ancestors", t.ArrayType(t.StringType()), True),])
    >>> disease_df = spark.createDataFrame(data, schema)
    +----------+-------------------------+
    |id        |ancestors                |
    +----------+-------------------------+
    |EFO_000001|[EFO_000001, EFO_0001444]|
    |EFO_000002|[EFO_000002]             |
    +----------+-------------------------+
    >>> lut = get_efo_ta_index(disease_df).show(truncate=False)
    +----------+----------------------+
    |id        |primaryTherapeuticArea|
    +----------+----------------------+
    |EFO_000001|EFO_0001444           |
    |EFO_000002|other                 |
    +----------+----------------------+

    """
    efo_ta = (
        disease.select("id", "ancestors")
        .withColumn("primaryTherapeuticArea", get_first_matching_therapeutic_area(f.col("ancestors"), ta_hierarchy))
        .withColumn(
            "primaryTherapeuticArea",
            f.when(f.col("primaryTherapeuticArea").isNull(), f.lit("other")).otherwise(f.col("primaryTherapeuticArea")),
        )
    )
    efo_ta_lookup = efo_ta.select("id", "primaryTherapeuticArea")
    return efo_ta_lookup


def assign_therapeutic_area_to_study_index(study: DataFrame, efo_ta_lookup: DataFrame) -> DataFrame:
    """Assign primary therapeutic areas to the diseaseIds in study."""
    # Explode the study
    exploded_study = study.select(f.col("studyId"), f.explode("diseaseIds").alias("diseaseIdsExploded"))

    # Left join the efo_ta_lookup
    # This results in [studyId, primaryTherapeuticAreas] rows
    annotated_study = (
        exploded_study.join(efo_ta_lookup, how="left", on=efo_ta_lookup.id == exploded_study.diseaseIdsExploded)
        .withColumn("primaryTherapeuticAreaName", _map_therapeutic_area_ids(f.col("primaryTherapeuticArea")))
        .drop("diseasIdsExploded", "primaryTherapeuticArea")
        .groupBy("studyId")
        .agg(f.collect_list("primaryTherapeuticAreaName").alias("mappedTherapeuticAreas"))
    )

    # Collect the therapeutic areas back to study
    return study.join(annotated_study, how="left", on="studyId")


def _map_therapeutic_area_ids(
    ta: Column, therapeutic_area_hierarchy: type[TherapeuticAreaHierarchy] = TherapeuticAreaHierarchy
) -> Column:
    """Map therapeutic area ids to names."""
    expr = f.when(f.lit(False), f.lit(None).cast(t.StringType()))

    for i in therapeutic_area_hierarchy:
        expr = expr.when(ta == f.lit(i.name), f.lit(i.value))

    # Fallback for `other`
    expr = expr.otherwise(f.lit("other"))
    return expr


def pivot_therapeutic_areas(study: DataFrame, ta_col: str = "mappedTherapeuticAreas") -> DataFrame:
    """Melt therapeutic areas column to separate columns.

    This operation converts long -> wide format.
    """
    exploded_study = (
        study.select(f.explode(ta_col).alias("ta"), f.col("studyId")).groupBy("studyId").pivot("ta").count()
    )
    columns_to_rename = [f.col(c).alias(string2camelcase(c)) for c in exploded_study.columns if c != "studyId"]
    columns_to_rename += [f.col("studyId")]

    ta_pivot = exploded_study.select(*columns_to_rename)
    return study.join(ta_pivot, on="studyId", how="left")

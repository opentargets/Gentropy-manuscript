"""Lead Variant Effect Dataset."""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
from enum import Enum

from gentropy.dataset.dataset import Dataset
from gentropy.dataset.study_locus import StudyLocus
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as f
from pyspark.sql.types import StructType

from manuscript_methods import schemas
from manuscript_methods.locus_statistics import LocusStatistics
from manuscript_methods.maf import MinorAlleleFrequency
from manuscript_methods.rescaled_beta import RescaledStatistics
from manuscript_methods.study_statistics import StudyStatistics, StudyType


class LVEControl(str, Enum):
    """Enum for study control types."""

    CASES_MORE_THEN_CONTROLS = "cases_more_than_controls"
    BETA_ABOVE_THRESHOLD = "beta_above_threshold"
    PREV_BELOW_THRESHOLD = "prev_below_threshold"
    N_SAMPLES_BELOW_THRESHOLD = "n_samples_below_threshold"


def parse_spark_schema(schema_json: str) -> StructType:
    """Parse Spark schema from JSON.

    Args:
        schema_json (str): JSON filename containing spark schema in the schemas package

    Returns:
        StructType: Spark schema

    """
    core_schema = json.loads(pkg_resources.read_text(schemas, schema_json, encoding="utf-8"))
    return StructType.fromJson(core_schema)


class LeadVariantEffect(Dataset):
    """Dataset for lead variant effect."""

    @classmethod
    def get_schema(cls) -> StructType:
        """Provide the schema for the LeadVariantEffect dataset.

        Returns:
            str: JSON string of the schema.

        """
        return parse_spark_schema("lead_variant_effect.json")

    def binary_trait_filter(
        self, prev_threshold=0.01, beta_threshold=3, n_samples_threshold=100_000
    ) -> LeadVariantEffect:
        """Filter the lead variant dataset to only include studies.

        Filtering applied on:
            * prevalence above or equal to threshold,
            * beta values below or equal to threshold,
            * number of samples above or equal threshold,
            * number of cases less than or equal number of controls,
            * prevelance exists
        """
        rescaled_stats = RescaledStatistics()
        study_stats = StudyStatistics()
        prev = rescaled_stats.prev
        beta = rescaled_stats.estimated_beta
        n_cases = study_stats.n_cases
        n_controls = study_stats.n_controls
        n_samples = study_stats.n_samples
        df = (
            self.df.filter(prev.isNotNull())
            .filter(n_cases <= n_controls)
            .filter(f.abs(beta) <= beta_threshold)
            .filter(n_samples >= n_samples_threshold)
            .filter(prev >= prev_threshold)
        )
        return LeadVariantEffect(df)

    def maf_filter(self, remove_null: bool = True, remove_zero: bool = True) -> LeadVariantEffect:
        """Filter out lead variants without calculated MAF."""
        maf = MinorAlleleFrequency(f.col("majorLdPopulationMaf")).value
        df = self.df
        if remove_zero:
            df = df.filter(maf > 0)
        if remove_null:
            df = df.filter(maf.isNotNull())
        return LeadVariantEffect(df)

    def effect_size_filter(self, effect_size_threshold: int = 3) -> LeadVariantEffect:
        """Filter lead variants based on effect size."""
        rescaled_stats = RescaledStatistics()
        beta = rescaled_stats.estimated_beta
        df = self.df.filter(f.abs(beta) <= effect_size_threshold)
        return LeadVariantEffect(df)

    def filter_by_study_locus_id(self, sl: StudyLocus) -> LeadVariantEffect:
        """Filter the dataset by study locus.

        Args:
            sl (StudyLocus): StudyLocus object containing the study locus data.

        Returns:
            LeadVariantEffect: Filtered dataset.

        """
        sl_dim = sl.df.select("studyLocusId").count()
        unique_sl_dim = sl.df.select("studyLocusId").distinct().count()
        print(f"StudyLocus dimension: {sl_dim}, unique studyLocusId: {unique_sl_dim}")
        if sl_dim != unique_sl_dim:
            raise ValueError("StudyLocus dataframe should have unique studyLocusId values.")
        df = self.df.join(sl.df.select("studyLocusId"), on=["studyLocusId"], how="inner")
        count_before = self.df.count()
        count_after = df.count()
        print(f"Initial rows: {count_before}")
        print(f"Filtered {count_before - count_after} rows based on the StudyLocus.")
        print(f"Remaining rows: {count_after}")
        return LeadVariantEffect(df)

    def replicated(self) -> DataFrame:
        """Filter only replicated credible sets.

        Filtering is based on following conditions and order:
        (1) pip >= 0.9
        (2) replicationCount >= 2
        (3) drop duplicates based on credibleSetId

        credibleSetId is calculated as the MD5 hash of the concatenation of variantId and traitId.
        The traitId is determined based on the study type:
        - For GWAS studies, it uses the concatenated and sorted traitFromSourceMappedIds
        - For other study types, it uses the molecular_trait field.

        """
        initial_n = self.df.count()
        study_stats = StudyStatistics()
        var_id = f.col("variantId")
        gwas_trait = f.concat_ws(",", f.array_sort(f.array_distinct(f.col("traitFromSourceMappedIds"))))
        trait_id = f.when(study_stats.study_type == StudyType.GWAS.value, gwas_trait).otherwise(
            study_stats.molecular_trait
        )
        credible_set_id = f.md5(f.concat_ws(",", var_id, trait_id, study_stats.study_type))
        # Add credible set Id as the concatenation of variantId and traitId and freeze the computation
        # to avoid recomputing it multiple times
        df = self.df.withColumn("credibleSetId", credible_set_id).persist()
        window = Window.partitionBy("credibleSetId")
        replication_count = f.count("credibleSetId").over(window)
        # Again make sure that the computation is done before filtering!
        df = df.withColumn("replicationCount", replication_count).persist()
        # apply filtering
        df = (
            df.filter(LocusStatistics().lead_variant_pip >= 0.9)
            .filter(f.col("replicationCount") >= 2)
            .dropDuplicates(["credibleSetId"])
        )
        following_n = df.count()
        print(f"Initial number of variants: {initial_n}")
        print(f"Following number of variants: {following_n}")
        print(f"Number of variants removed: {initial_n - following_n}")
        print(f"Percentage of variants removed: {(initial_n - following_n) / initial_n:.2%}")
        return df

    def limit(self) -> LeadVariantEffect:
        """Limit the dataset to only cis-pQTL, GWAS and eQTL studies."""
        df = self.df.filter(
            f.col("studyStatistics.studyTYpe").isin(
                [StudyType.CIS_PQTL.value, StudyType.GWAS.value, StudyType.EQTL.value]
            )
        )
        return LeadVariantEffect(df)

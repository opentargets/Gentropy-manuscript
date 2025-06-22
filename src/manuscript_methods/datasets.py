"""Lead Variant Effect Dataset."""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
from enum import Enum

from gentropy.dataset.dataset import Dataset
from pyspark.sql import Column
from pyspark.sql import functions as f
from pyspark.sql.types import StructType

from manuscript_methods import schemas
from manuscript_methods.rescaled_beta import RescaledStatistics
from manuscript_methods.study_statistics import StudyStatistics
from manuscript_methods.maf import MinorAlleleFrequency


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

    def lve_filter(self, prev_threshold=0.01, beta_threshold=3, n_samples_threshold=100_000) -> LeadVariantEffect:
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

    def maf_filter(self) -> LeadVariantEffect:
        """Filter out lead variants without calculated MAF."""
        maf = MinorAlleleFrequency().value
        df = self.df.filter(maf.isNotNull())
        return LeadVariantEffect(df)

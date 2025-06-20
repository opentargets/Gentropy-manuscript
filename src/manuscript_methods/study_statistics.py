"""Cohort definitions."""

from __future__ import annotations

from enum import Enum

from pyspark.sql import Column
from pyspark.sql import functions as f


class StudyType(str, Enum):
    """Enum for study types."""

    GWAS = "gwas"
    EQTL = "eqtl"
    SCEQTL = "sceqtl"
    PQTL = "cis-pqtl"
    SQTL = "sqtl"
    TUQTL = "tuqtl"
    TRANS_PQTL = "trans-pqtl"
    CIS_PQTL = "cis-pqtl"


class CaseControlDiscrepancy(str, Enum):
    """Case control discrepancy types."""

    EMPTY_CASES_NON_EMPTY_CONTROLS = "emptyCasesNonEmptyControls"
    EMPTY_CONTROLS_NON_EMPTY_CASES = "emptyControlsNonEmptyCases"
    SUM_CASES_CONTROLS_NEQUAL_SAMPLES = "sumCasesControlsNotEqualSamples"


class TraitClassName(str, Enum):
    """Enum for trait class names."""

    QUANTITATIVE = "quantitative"
    BINARY = "binary"
    UNKNOWN = "unknown"


class StudyStatistics:
    """Study class to define a study statistics."""

    name = "studyStatistics"
    """Name of the study statistics."""
    schema = "struct<nCases: INT, nControls: INT, nSamples: INT, trait: STRING, studyType: STRING, traitClass: STRING, traitFromSourceMappedIds: ARRAY<STRING>>"

    non_gwas_study_types = [
        StudyType.EQTL.value,
        StudyType.SCEQTL.value,
        StudyType.PQTL.value,
        StudyType.SQTL.value,
        StudyType.TUQTL.value,
    ]

    def __init__(self, col: Column | None = None):
        """Initialize Cohort with an optional column.

        Args:
            col (Column, optional): Optional column to initialize the cohort.


        """
        self.col = col.alias(self.name) if col is not None else f.col(self.name)

    @property
    def n_cases(self) -> Column:
        """Get the number of cases in the cohort."""
        return self.col.getField("nCases").alias("nCases")

    @property
    def n_controls(self) -> Column:
        """Get the number of controls in the cohort."""
        return self.col.getField("nControls").alias("nControls")

    @property
    def n_samples(self) -> Column:
        """Get the total number of samples in the cohort."""
        return self.col.getField("nSamples").alias("nSamples")

    @property
    def trait(self) -> Column:
        """Get the trait associated with the cohort."""
        return self.col.getField("trait").alias("trait")

    @property
    def study_type(self) -> Column:
        """Get the study type associated with the cohort."""
        return self.col.getField("studyType").alias("studyType")

    @property
    def trait_class(self) -> Column:
        """Get the trait class associated with the cohort."""
        return self.col.getField("traitClass").alias("traitClass")

    @property
    def gene_id(self) -> Column:
        """Get the gene ID associated with the cohort."""
        return self.col.getField("geneId").alias("geneId")

    @property
    def trait_ids(self) -> Column:
        """Get the trait IDs associated with the cohort."""
        return self.col.getField("traitFromSourceMappedIds").alias("traitFromSourceMappedIds")

    @classmethod
    def classify_trait(cls, trait: Column, n_cases: Column, n_controls: Column, study_type: Column) -> Column:
        """Classify the trait as continuous or binary."""
        expr = (
            f.when(study_type.isin(cls.non_gwas_study_types), f.lit(TraitClassName.QUANTITATIVE))
            .when((trait.isNull()) | (trait == f.lit("")), f.lit(TraitClassName.UNKNOWN))
            .when((n_cases > 0) & (n_controls > 0), f.lit(TraitClassName.BINARY))
            .when((n_cases == 0) & (n_controls == 0), f.lit(TraitClassName.QUANTITATIVE))
        )

        return expr.alias("traitClass")

    @classmethod
    def split_pqtl(cls, study_type: Column, is_trans_pqtl: Column) -> Column:
        """Transform the study type to a string."""
        expr = (
            f.when((study_type == f.lit(StudyType.PQTL)) & is_trans_pqtl, f.lit(StudyType.TRANS_PQTL))
            .when((study_type == f.lit(StudyType.PQTL)) & ~is_trans_pqtl, f.lit(StudyType.CIS_PQTL))
            .otherwise(study_type)
        )

        return expr.alias("studyType")

    def validate_trait_class(self) -> Column:
        """Validate the trait class."""
        expr = (
            f.when((self.n_cases > 0) & (self.n_controls == 0), CaseControlDiscrepancy.EMPTY_CONTROLS_NON_EMPTY_CASES)
            .when((self.n_cases == 0) & (self.n_controls > 0), CaseControlDiscrepancy.EMPTY_CASES_NON_EMPTY_CONTROLS)
            .when(
                (self.n_cases + self.n_controls) != self.n_samples,
                CaseControlDiscrepancy.SUM_CASES_CONTROLS_NEQUAL_SAMPLES,
            )
        )

        return expr.alias("caseControlDiscrepancy")

    @classmethod
    def compute(
        cls,
        n_cases: Column,
        n_controls: Column,
        n_samples: Column,
        trait: Column,
        study_type: Column,
        trait_ids: Column,
        is_trans_pqtl: Column,
        gene_id: Column,
    ) -> StudyStatistics:
        """Compute the cohort statistics from the number of cases, controls, and trait.

        Args:
            n_cases (Column): Number of cases in the cohort.
            n_controls (Column): Number of controls in the cohort.
            n_samples (Column): Total number of samples in the cohort.
            trait (Column): Trait associated with the cohort.

        Returns:
            studyStatistics: A studyStatistics object containing the computed cohort statistics.

        Examples:
        --------

        """
        return cls(
            f.struct(
                n_cases.alias("nCases"),
                n_controls.alias("nControls"),
                n_samples.alias("nSamples"),
                gene_id.alias("geneId"),
                trait.alias("trait"),
                cls.split_pqtl(study_type, is_trans_pqtl),
                cls.classify_trait(trait, n_cases, n_controls, study_type),
                trait_ids.alias("traitFromSourceMappedIds"),
            )
        )

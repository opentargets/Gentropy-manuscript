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
    PQTL = "pqtl"
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
    schema = "struct<nCases: INT, nControls: INT, nSamples: INT, trait: STRING, studyType: STRING, traitClass: STRING>"

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
    def molecular_trait(self) -> Column:
        """Get the gene ID associated with the cohort."""
        return self.col.getField("molecularTrait").alias("molecularTrait")

    @classmethod
    def classify_trait(cls, n_cases: Column, n_controls: Column, study_type: Column) -> Column:
        """Classify the trait as continuous or binary."""
        expr = (
            f.when(study_type.isin(cls.non_gwas_study_types), f.lit(TraitClassName.QUANTITATIVE))
            .when((n_cases > 0) & (n_controls > 0), f.lit(TraitClassName.BINARY))
            .when((n_cases == 0), f.lit(TraitClassName.QUANTITATIVE))
            .when((n_cases.isNull()), f.lit(TraitClassName.QUANTITATIVE))
            .when((n_controls == 0), f.lit(TraitClassName.QUANTITATIVE))
            .when((n_controls.isNull()), f.lit(TraitClassName.QUANTITATIVE))
            .otherwise(f.lit(TraitClassName.UNKNOWN))
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

    @classmethod
    def merge_gwas_and_molecular_traits(cls, gene_id: Column, trait: Column) -> Column:
        """Merge GWAS and molecular traits into a single trait column."""
        return f.coalesce(trait, gene_id).alias("trait")

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
        is_trans_pqtl: Column,
        gene_id: Column,
    ) -> StudyStatistics:
        """Compute the cohort statistics from the number of cases, controls, and trait.

        The cardinality of this table is 1:1 with credible sets.

        Args:
            n_cases (Column): Number of cases in the cohort.
            n_controls (Column): Number of controls in the cohort.
            n_samples (Column): Total number of samples in the cohort.
            trait (Column): Trait associated with the cohort.
            study_type (Column): Type of study (e.g., gwas, eqtl, etc.).
            is_trans_pqtl (Column): Boolean indicating if the credible set refers to cis or trans qtl.
            gene_id (Column): Gene ID associated with the molecular trait.

        Returns:
            studyStatistics: A studyStatistics object containing the computed cohort statistics.

        Examples:
        --------
        >>> r1 = (100, 50, 150, "EFO_0000508", "gwas", False, None)
        >>> r2 = (0, 0, 210, "EFO_0000408", "pqtl", True, "ENSG00000139618")
        >>> r3 = (None, None, 300, "EFO_0000608", "gwas", False, None)
        >>> data = [r1, r2, r3]
        >>> schema = "nCases INT, nControls INT, nSamples INT, trait STRING, studyType STRING, isTransPqtl BOOLEAN, geneId STRING"
        >>> df = spark.createDataFrame(data, schema)
        >>> df.show()
        +------+---------+--------+-----------+---------+-----------+---------------+
        |nCases|nControls|nSamples|      trait|studyType|isTransPqtl|         geneId|
        +------+---------+--------+-----------+---------+-----------+---------------+
        |   100|       50|     150|EFO_0000508|     gwas|      false|           NULL|
        |     0|        0|     210|EFO_0000408|     pqtl|       true|ENSG00000139618|
        |  NULL|     NULL|     300|EFO_0000608|     gwas|      false|           NULL|
        +------+---------+--------+-----------+---------+-----------+---------------+
        <BLANKLINE>
        >>> study_stats = StudyStatistics.compute(
        ... n_cases=f.col("nCases"),
        ... n_controls=f.col("nControls"),
        ... n_samples=f.col("nSamples"),
        ... trait=f.col("trait"),
        ... study_type=f.col("studyType"),
        ... is_trans_pqtl=f.col("isTransPqtl"),
        ... gene_id=f.col("geneId"),
        ... )
        >>> df = df.select(study_stats.col)
        >>> df.select("studyStatistics.*").show()
        +------+---------+--------+-----------+----------+------------+---------------+
        |nCases|nControls|nSamples|      trait| studyType|  traitClass| molecularTrait|
        +------+---------+--------+-----------+----------+------------+---------------+
        |   100|       50|     150|EFO_0000508|      gwas|      binary|           NULL|
        |     0|        0|     210|EFO_0000408|trans-pqtl|quantitative|ENSG00000139618|
        |  NULL|     NULL|     300|EFO_0000608|      gwas|quantitative|           NULL|
        +------+---------+--------+-----------+----------+------------+---------------+
        <BLANKLINE>

        """
        return cls(
            f.struct(
                n_cases.alias("nCases"),
                n_controls.alias("nControls"),
                n_samples.alias("nSamples"),
                cls.merge_gwas_and_molecular_traits(gene_id, trait).alias("trait"),
                cls.split_pqtl(study_type, is_trans_pqtl),
                cls.classify_trait(n_cases, n_controls, study_type),
                f.col("geneId").alias("molecularTrait"),
            )
        )

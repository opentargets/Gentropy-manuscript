"""MAF related methods."""

from collections.abc import Callable
from enum import Enum

from pyspark.sql import Column
from pyspark.sql import functions as f

from manuscript_methods.ld_populations import LDPopulation, LDPopulationName


class AlleleFrequencyPopulationName(str, Enum):
    """Enum representing allele frequency population names.

    Based on gnomAD v4.1 allele frequency population names.
    """

    AFR = "afr_adj"
    AMR = "amr_adj"
    EAS = "eas_adj"
    NFE = "nfe_adj"
    FIN = "fin_adj"

    # Other populations that we do not know how to convert to LDPopulationName
    REMAINING = "remaining_adj"
    MID = "mid_adj"
    AMI = "ami_adj"
    SAS = "sas_adj"
    ASJ = "asj_adj"


class PopulationConverter:
    """Class to convert population names to allele frequency population names."""

    @staticmethod
    def ld_to_af(population: Column) -> Column:
        """Convert LDPopulationName to AlleleFrequencyPopulationName."""
        return f.concat(population, f.lit("_adj")).alias("AFFromLDPopulationName")

    @staticmethod
    def af_to_ld(population: Column) -> Column:
        """Convert AlleleFrequencyPopulationName to LDPopulationName."""
        raise NotImplementedError(
            "Conversion from AlleleFrequencyPopulationName to LDPopulationName is not implemented."
        )


class MAFDiscrepancies(str, Enum):
    """Enum representing MAF discrepancies."""

    MAJOR_ANCESTRY_MISSING_FROM_GNOMAD_AF = "MAJOR_ANCESTRY_MISSING_FROM_GNOMAD_AF"
    VARIANT_MISSING_FROM_GNOMAD_AF = "VARIANT_MISSING_FROM_GNOMAD_AF"


class AlleleFrequency:
    """Class representing allele frequency."""

    schema = "STRUCT<populationName: STRING, alleleFrequency: DOUBLE>"

    def __init__(self, col: Column | None = None):
        """Initialize AlleleFrequency with a column."""
        self.name = "alleleFrequency"
        self.col = col.alias(self.name) if col is not None else f.col(self.name)

    @property
    def population_name(self) -> Column:
        """Get population name from allele frequency."""
        return self.col.getField("populationName").alias("populationName")

    @property
    def allele_frequency(self) -> Column:
        """Get allele frequency from allele frequency."""
        return self.col.getField("alleleFrequency").alias("alleleFrequency")

    def maf(self) -> Column:
        """Calculate Minor Allele Frequency from variant frequency."""
        return (
            f.when((self.allele_frequency > 0.5), f.lit(1.0) - self.allele_frequency)
            .when((self.allele_frequency <= 0.5), self.allele_frequency)
            .otherwise(None)
        )


class AlleleFrequencies:
    """Class representing a collection of allele frequencies."""

    schema = f"ARRAY<{AlleleFrequency.schema}>"

    def __init__(self, col: Column | None = None):
        """Initialize AlleleFrequencies with a column."""
        self.name = "alleleFrequencies"
        self.col = col.alias(self.name) if col is not None else f.col(self.name)

    def major_ld_population_frequency(self, population_name: Column) -> AlleleFrequency:
        """Get allele frequency for a specific population."""
        ld_population = PopulationConverter.ld_to_af(population_name)
        _filter: Callable[[Column], Column]
        _filter = lambda x: x.getField("populationName") == ld_population

        return AlleleFrequency(f.filter(self.col, _filter))

    def major_ld_population_maf(self, population_name: Column) -> Column:
        """Calculate Minor Allele Frequency from variant frequency."""
        major_population_af = self.major_ld_population_frequency(population_name)
        expr = f.when(f.size(major_population_af.col) == 1, major_population_af.maf()).otherwise(None)
        return expr.alias("majorPopulationMAF")


def maf_discrepancies(maf: Column) -> Column:
    """Add maf discrepancies.

    The method adds the column `mafDiscrepancy` to the DataFrame, which contains
    information about discrepancies in the minor allele frequency (MAF) calculation.

    The possible discrepancies are:
    - `MAJOR_ANCESTRY_MISSING_FROM_GNOMAD_AF`: The major ancestry is missing from gnomAD allele frequency table, variant was captured by GnomAD.
    - `VARIANT_MISSING_FROM_GNOMAD_AF`: The variant is missing from gnomAD allele frequency, variant was not captured by GnomAD.

    """
    expr = (
        f.when(maf.isNull(), f.lit(MAFDiscrepancies.VARIANT_MISSING_FROM_GNOMAD_AF))
        .when(maf == 0.0, f.lit(MAFDiscrepancies.MAJOR_ANCESTRY_MISSING_FROM_GNOMAD_AF))
        .otherwise(None)
    )
    return expr.alias("mafDiscrepancy")

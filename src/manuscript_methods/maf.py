"""MAF related methods."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from pyspark.sql import Column
from pyspark.sql import functions as f


class MinorAlleleFrequencyType(str, Enum):
    """Enum representing minor allele frequency types.

    The enum is used to distinguish between different types of minor allele frequencies.
    """

    NOT_FLIPPED = "notFlipped"
    FLIPPED = "flipped"
    AMBIGUOUS = "ambiguous"


class MinorAlleleFrequency:
    """Class representing minor allele frequency."""

    name = "minorAlleleFrequency"
    schema = "STRUCT<value: DOUBLE, type: STRING>"

    def __init__(self, col: Column | None = None):
        """Initialize MinorAlleleFrequency with a column.

        Args:
            col (Column | None): Column representing minor allele frequency. If None, a default column will
            be created with the name 'minorAlleleFrequency'.

        """
        self.col = col.alias(self.name) if col is not None else f.col(self.name)

    @property
    def value(self) -> Column:
        """Get value from minor allele frequency."""
        return self.col.getField("value").alias("value")

    @property
    def type(self) -> Column:
        """Get type from minor allele frequency."""
        return self.col.getField("type").alias("type")

    @classmethod
    def from_af(cls, population_frequency: PopulationFrequency) -> MinorAlleleFrequency:
        """Calculate Minor Allele Frequency from allele frequency."""
        af = population_frequency.allele_frequency
        flipped = f.struct((f.lit(1.0) - af).alias("value"), f.lit("flipped").alias("type"))
        non_flipped = f.struct(af.alias("value"), f.lit("notFlipped").alias("type"))
        ambiguous = f.struct(af.alias("value"), f.lit("ambiguous").alias("type"))

        return cls(
            f.when(af > 0.5, flipped)
            .when(af < 0.5, non_flipped)
            .when(af == 0.5, ambiguous)  # Handle the case where allele frequency is exactly 0.5
            .otherwise(None)
        )


class AlleleFrequencyPopulationName(str, Enum):
    """Enum representing allele frequency population names.

    Based on gnomAD v4.1 allele frequency population names.
    """

    AFR = "afr_adj"
    AMR = "amr_adj"
    EAS = "eas_adj"
    NFE = "nfe_adj"
    FIN = "fin_adj"

    # Other populations that we do not have currently the LD for.
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

    MAJOR_ANCESTRY_MISSING_FROM_GNOMAD_AF = "Variant missing from major ancestry in gnomAD AF"
    VARIANT_MISSING_FROM_GNOMAD_AF = "Variant missing from any ancestry in gnomAD AF"
    VARIANT_HAS_AF = "Variant has allele frequency in gnomAD AF"


class PopulationFrequency:
    """Class representing allele frequency."""

    schema = "STRUCT<populationName: STRING, alleleFrequency: DOUBLE>"
    name = "populationFrequency"

    def __init__(self, col: Column | None = None):
        """Initialize PopulationFrequency with a column.

        Args:
            col (Column | None): Column representing allele frequency. If None, a default column will
            be created with the name 'populationFrequency'.

        Examples:
        --------
        >>> data = [(("afr_adj", 0.1),), (("amr_adj", 0.2),)]
        >>> schema = f"{PopulationFrequency.name}: {PopulationFrequency.schema}"
        >>> print(schema)
        populationFrequency: STRUCT<populationName: STRING, alleleFrequency: DOUBLE>
        >>> df = spark.createDataFrame(data, schema)
        >>> af = PopulationFrequency()
        >>> af.population_name
        Column<'populationFrequency[populationName] AS populationName'>
        >>> af.allele_frequency
        Column<'populationFrequency[alleleFrequency] AS alleleFrequency'>
        >>> df.show()
        +-------------------+
        |populationFrequency|
        +-------------------+
        |     {afr_adj, 0.1}|
        |     {amr_adj, 0.2}|
        +-------------------+
        <BLANKLINE>

        """
        self.col = col.alias(self.name) if col is not None else f.col(self.name)

    @property
    def population_name(self) -> Column:
        """Get population name from allele frequency."""
        return self.col.getField("populationName").alias("populationName")

    @property
    def allele_frequency(self) -> Column:
        """Get allele frequency from allele frequency."""
        return self.col.getField("alleleFrequency").alias("alleleFrequency")

    @property
    def maf(self) -> MinorAlleleFrequency:
        """Calculate Minor Allele Frequency from variant frequency.

        Returns:
            Column: Minor Allele Frequency column.

        Examples:
        --------
        >>> data = [(("AFR", 0.1),), (("AMR", 0.9),), (("EAS", 0.5),)]
        >>> schema = f"{PopulationFrequency.name}: {PopulationFrequency.schema}"
        >>> df = spark.createDataFrame(data, schema)
        >>> maf = PopulationFrequency().maf.col
        >>> df = df.withColumn("minorAlleleFrequency", maf)
        >>> af = f.col("populationFrequency")
        >>> maf_value = f.round(maf.getField("value"), 1).alias("value")
        >>> conversion_type = maf.getField("type").alias("type")
        >>> df = df.select(af, maf_value, conversion_type)
        >>> df.show(truncate=False)
        +-------------------+-----+----------+
        |populationFrequency|value|type      |
        +-------------------+-----+----------+
        |{AFR, 0.1}         |0.1  |notFlipped|
        |{AMR, 0.9}         |0.1  |flipped   |
        |{EAS, 0.5}         |0.5  |ambiguous |
        +-------------------+-----+----------+
        <BLANKLINE>

        """
        return MinorAlleleFrequency.from_af(self)


class AlleleFrequencies:
    """Class representing a collection of allele frequencies."""

    schema = f"ARRAY<{PopulationFrequency.schema}>"
    name = "alleleFrequencies"

    def __init__(self, col: Column | None = None):
        """Initialize AlleleFrequencies with a column.

        Args:
            col (Column | None): Column representing allele frequencies. If None, a default column will
            be created with the name 'alleleFrequencies'.

        Examples:
        --------
        >>> x1 = [("AFR", 0.1), ("AMR", 0.2)]
        >>> x2 = [("EAS", 0.3), ("NFE", 0.4)]
        >>> data = [(x1,), (x2,)]
        >>> schema = f"{AlleleFrequencies.name}: {AlleleFrequencies.schema}"
        >>> df = spark.createDataFrame(data, schema)
        >>> af = AlleleFrequencies()
        >>> df.select(af.col).show(truncate=False)
        +------------------------+
        |alleleFrequencies       |
        +------------------------+
        |[{AFR, 0.1}, {AMR, 0.2}]|
        |[{EAS, 0.3}, {NFE, 0.4}]|
        +------------------------+
        <BLANKLINE>

        """
        self.col = col.alias(self.name) if col is not None else f.col(self.name)

    def ld_population_af(self, population_name: Column) -> PopulationFrequency:
        """Get allele frequency for a specific population.

        Args:
            population_name (Column): Column representing the ld population name.

        Returns:
            PopulationFrequency: PopulationFrequency object containing allele frequency for the specified population.

        Examples:
        --------
        >>> x1 = [("afr_adj", 0.1), ("amr_adj", 0.2)]
        >>> x2 = [("eas_adj", 0.3), ("nfe_adj", 0.4)]
        >>> data = [(x1, "afr"), (x2, "eas")]
        >>> schema = f"{AlleleFrequencies.name}: {AlleleFrequencies.schema}, populationName: STRING"
        >>> df = spark.createDataFrame(data, schema)
        >>> ld_af = AlleleFrequencies().ld_population_af(f.col("populationName"))
        >>> df.select(ld_af.col).show(truncate=False)
        +-------------------+
        |populationFrequency|
        +-------------------+
        |{afr_adj, 0.1}     |
        |{eas_adj, 0.3}     |
        +-------------------+
        <BLANKLINE>

        """
        ld_population = PopulationConverter.ld_to_af(population_name)
        _filter: Callable[[Column], Column]
        _filter = lambda x: x.getField("populationName") == ld_population
        ld_pop_or_empty = f.filter(self.col, _filter)
        expr = (
            f.when(f.size(ld_pop_or_empty) == 1, ld_pop_or_empty.getItem(0))
            .otherwise(None)
            .alias("populationFrequency")
        )

        return PopulationFrequency(expr)

    def ld_population_maf(self, population_name: Column) -> MinorAlleleFrequency:
        """Calculate Minor Allele Frequency from variant frequency.

        Args:
            population_name (Column): Column representing the ld population name.

        Returns:
            MinorAlleleFrequency: Class representing column containing the major population MAF.

        Examples:
        --------
        >>> x1 = [("afr_adj", 0.1), ("amr_adj", 0.2)]
        >>> x2 = [("eas_adj", 0.3), ("nfe_adj", 0.4)]
        >>> data = [(x1, "afr"), (x2, "eas")]
        >>> schema = f"{AlleleFrequencies.name}: {AlleleFrequencies.schema}, populationName: STRING"
        >>> df = spark.createDataFrame(data, schema)
        >>> ld_af = AlleleFrequencies().ld_population_maf(f.col("populationName"))
        >>> df.select(ld_af).show(truncate=False)
        +------------------+
        |majorPopulationMaf|
        +------------------+
        |{0.1, notFlipped} |
        |{0.3, notFlipped} |
        +------------------+
        <BLANKLINE>

        """
        ld_population_af = self.ld_population_af(population_name)
        return ld_population_af.maf


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
        .otherwise(MAFDiscrepancies.VARIANT_HAS_AF)
    )
    return expr.alias("mafDiscrepancy")

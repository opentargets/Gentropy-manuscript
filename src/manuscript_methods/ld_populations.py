"""LD populations."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from pyspark.sql import Column
from pyspark.sql import functions as f


class LDPopulationName(str, Enum):
    """Enum representing LD populations."""

    AFR = "afr"
    AMR = "amr"
    EAS = "eas"
    NFE = "nfe"
    FIN = "fin"


class LDPopulation:
    """Class representing a single LD population.

    This class represents a column containing the:
    * `ldPopulation` - name of the LD population
    * `relativeSampleSize` - relative sample size of the LD population.

    Examples
    --------
    **Create from components (column expression):**
    >>> ld_population = LDPopulation.from_components(LDPopulationName.NFE, 0.9)
    >>> print(ld_population.col)
    Column<'struct(nfe AS ldPopulation, 0.9 AS relativeSampleSize) AS ldPopulation'>

    **Create from default column:**
    >>> ld_population = LDPopulation()
    >>> print(ld_population.col)
    Column<'ldPopulation'>

    **Get the column name and schema:**
    >>> print(ld_population.name)
    ldPopulation
    >>> print(ld_population.schema)
    STRUCT<ldPopulation: STRING, relativeSampleSize: DOUBLE>

    **Get the relative sample size and LD population nested columns:**
    >>> print(ld_population.relative_sample_size)
    Column<'ldPopulation[relativeSampleSize]'>
    >>> print(ld_population.ld_population)
    Column<'ldPopulation[ldPopulation]'>

    **Check if the LD population is a specific population:**
    >>> print(ld_population.is_population(LDPopulationName.NFE))
    Column<'(ldPopulation[ldPopulation] = nfe)'>

    """

    schema = "STRUCT<ldPopulation: STRING, relativeSampleSize: DOUBLE>"

    def __init__(self, col: Column | None = None):
        self.name: str = "ldPopulation"
        self.col = col.alias(self.name) if col is not None else f.col(self.name)

    @property
    def relative_sample_size(self) -> Column:
        """Get the relative sample size from the LD population structure."""
        return self.col.getField("relativeSampleSize")

    @property
    def ld_population(self) -> Column:
        """Get the LD population from the LD population structure."""
        return self.col.getField("ldPopulation")

    def is_population(self, population: LDPopulationName) -> Column:
        """Check if the LD population matches a specific population."""
        return self.ld_population == f.lit(population.value)

    def __gt__(self, other: LDPopulation) -> Column:
        """Compare two LD populations based on relative sample size."""
        return self.relative_sample_size > other.relative_sample_size

    def __lt__(self, other: LDPopulation) -> Column:
        """Compare two LD populations based on relative sample size."""
        return self.relative_sample_size < other.relative_sample_size

    def __eq__(self, other: LDPopulation) -> Column:  # type: ignore
        """Check if two LD populations are equal based on relative sample size."""
        return self.relative_sample_size == other.relative_sample_size

    def __ne__(self, other: LDPopulation) -> Column:  # type: ignore
        """Check if two LD populations are not equal based on relative sample size."""
        return self.relative_sample_size != other.relative_sample_size

    def __le__(self, other: LDPopulation) -> Column:
        """Check if this LD population is less than or equal to another based on relative sample size."""
        return self.relative_sample_size <= other.relative_sample_size

    def __ge__(self, other: LDPopulation) -> Column:
        """Check if this LD population is greater than or equal to another based on relative sample size."""
        return self.relative_sample_size >= other.relative_sample_size

    @classmethod
    def from_components(
        cls, ld_population_name: LDPopulationName = LDPopulationName.NFE, relative_sample_size: float = 0.0
    ) -> LDPopulation:
        """Create LDPopulation from LDPopulationName and relative sample size."""
        return cls(
            f.struct(
                f.lit(ld_population_name.value).alias("ldPopulation"),
                f.lit(relative_sample_size).alias("relativeSampleSize"),
            )
        )


class LDPopulationStructure:
    """Study representation of the LD population structure.

    This class represents the LD population structure in a study, which is an array of
    structures containing the LD population name and its relative sample size.
    The structure is defined as an array of structs with the following fields:
    - ldPopulation: STRING - The name of the LD population.
    - relativeSampleSize: DOUBLE - The relative sample size of the LD population.


    Examples
    --------
    >>> x1 = [('nfe',0.900),('eas',0.018),('afr',0.082)]
    >>> x2 = [('nfe',1.000),]
    >>> x3 = [('eas',0.500),('nfe',0.500)]
    >>> data = [(x1,),(x2,),(x3,)]
    >>> schema = f"ldPopulationStructure: {LDPopulationStructure.schema}"
    >>> df = spark.createDataFrame(data, schema=schema)
    >>> ld = LDPopulationStructure()
    >>> df.show(truncate=False)
    +----------------------------------------+
    |ldPopulationStructure                   |
    +----------------------------------------+
    |[{nfe, 0.9}, {eas, 0.018}, {afr, 0.082}]|
    |[{nfe, 1.0}]                            |
    |[{eas, 0.5}, {nfe, 0.5}]                |
    +----------------------------------------+
    <BLANKLINE>

    **Get the column name**
    >>> print(ld.name)
    ldPopulationStructure

    **Get the schema of the LD population structure column**
    >>> print(ld.schema)
    ARRAY<STRUCT<ldPopulation: STRING, relativeSampleSize: DOUBLE>>

    **Get the column representing the LD population structure**
    >>> ld.col
    Column<'ldPopulationStructure'>

    **Construct with an expression**
    >>> ld_expr = f.struct(
    ...     f.array(
    ...         f.struct(f.lit("nfe").alias("ldPopulation"), f.lit(0.9).alias("relativeSampleSize")),
    ...     )
    ... )
    >>> ld = LDPopulationStructure(ld_expr)
    >>> print(ld.col)
    Column<'struct(array(struct(nfe AS ldPopulation, 0.9 AS relativeSampleSize))) AS ldPopulationStructure'>

    """

    schema = f"ARRAY<{LDPopulation.schema}>"

    def __init__(self, col: Column | None = None):
        self.name: str = "ldPopulationStructure"
        self.col = col.alias(self.name) if col is not None else f.col(self.name)

    @property
    def size(self) -> Column:
        """Get the size of the LD population structure.

        Returns:
            Column: `ldPopulationStructure` column representing the size of the LD population structure.

        Examples:
        --------
        >>> x1 = [('nfe',0.900),('eas',0.018),('afr',0.082)]
        >>> x2 = [('nfe',1.000),]
        >>> x3 = [('eas',0.500),('nfe',0.500)]
        >>> data = [(x1,),(x2,),(x3,)]
        >>> ld = LDPopulationStructure()
        >>> schema = f"ldPopulationStructure: {ld.schema}"
        >>> df = spark.createDataFrame(data, schema=schema)
        >>> df.show(truncate=False)
        +----------------------------------------+
        |ldPopulationStructure                   |
        +----------------------------------------+
        |[{nfe, 0.9}, {eas, 0.018}, {afr, 0.082}]|
        |[{nfe, 1.0}]                            |
        |[{eas, 0.5}, {nfe, 0.5}]                |
        +----------------------------------------+
        <BLANKLINE>
        >>> df.select(ld.size).show(truncate=False)
        +-------------------------+
        |ldPopulationStructureSize|
        +-------------------------+
        |3                        |
        |1                        |
        |2                        |
        +-------------------------+
        <BLANKLINE>

        """
        return f.size(self.col).alias("ldPopulationStructureSize")

    @staticmethod
    def find_larger_ld_population(
        pop1: LDPopulation, pop2: LDPopulation, default_major_pop: LDPopulationName
    ) -> LDPopulation:
        """Compare two populations based on relative sample size and get the population with the larger relative sample size.

        This function takes 2 populations and report one of them based on following conditions:
        * Use pop with bigger relativeSampleSize
        * In case of a tie, the default_major_pop is preferred,
        * In case of tie and no default_major_pop in pop1 and pop2, use pop1.
        * In case of no populations present, use default_major_pop with relativeSampleSize 0.0.

        Args:
            pop1 (LDPopulation): The first LD population to compare.
            pop2 (LDPopulation): The second LD population to compare.
            default_major_pop (LDPopulationName): The default major population to use when no populations are present or in case of a tie.

        Returns:
            LDPopulation: The larger LD population based on the relative sample size, or the default major population if no populations are present.

        Examples:
        --------
        >>> data = [(('nfe', 0.9), ('eas', 0.018),),]
        >>> ld_schema = LDPopulation().schema
        >>> schema = "pop1: struct<ldPopulation:string,relativeSampleSize:double>, pop2: struct<ldPopulation:string,relativeSampleSize:double>"
        >>> df = spark.createDataFrame(data, schema=schema)
        >>> df.show(truncate=False)
        +----------+------------+
        |pop1      |pop2        |
        +----------+------------+
        |{nfe, 0.9}|{eas, 0.018}|
        +----------+------------+
        <BLANKLINE>
        >>> larger_pop = LDPopulationStructure.find_larger_ld_population(
        ...    pop1=LDPopulation(f.col("pop1")),
        ...    pop2=LDPopulation(f.col("pop2")),
        ...    default_major_pop=LDPopulationName.NFE,
        ... )
        >>> df.select(larger_pop.col.alias("largerPop")).show(truncate=False)
        +----------+
        |largerPop |
        +----------+
        |{nfe, 0.9}|
        +----------+
        <BLANKLINE>

        """
        return LDPopulation(
            f.when(pop1 > pop2, pop1.col)
            .when(pop1 < pop2, pop2.col)
            .when(
                ((pop1 == pop2) & (pop1.is_population(default_major_pop))),
                pop1.col,
            )
            .when(
                ((pop1 == pop2) & (pop2.is_population(default_major_pop))),
                pop2.col,
            )
            .otherwise(pop1.col)
        )

    def major_population(self, default_major_pop: LDPopulationName) -> LDPopulation:
        """Extract the major population from the study ld population structure.

        Args:
            default_major_pop (LDPopulationName): The default major population to use when no populations are present or in case of a tie.

        Returns:
            LDPopulation: The major population with the largest relative sample size, or the default major population if no populations are present.

        Examples:
        --------
        >>> x1 = [('nfe',0.900),('eas',0.018),('afr',0.082)] # nfe choosen as it is largest population
        >>> x2 = [('nfe',1.000),]                            # nfe choosen as it is the only population
        >>> x3 = [('eas',0.500),('nfe',0.500)]               # nfe choosen when tie and nfe is default
        >>> x4 = [('eas',0.500),('afr',0.500)]               # eas choosen when tie, first in the list
        >>> x5 = []                                          # empty array, assume nfe with 0.0 sample
        >>> data = [(x1,),(x2,),(x3,),(x4,),(x5,)]
        >>> ld = LDPopulationStructure()
        >>> schema = f"ldPopulationStructure: {ld.schema}"
        >>> df = spark.createDataFrame(data, schema=schema)
        >>> df.show(truncate=False)
        +----------------------------------------+
        |ldPopulationStructure                   |
        +----------------------------------------+
        |[{nfe, 0.9}, {eas, 0.018}, {afr, 0.082}]|
        |[{nfe, 1.0}]                            |
        |[{eas, 0.5}, {nfe, 0.5}]                |
        |[{eas, 0.5}, {afr, 0.5}]                |
        |[]                                      |
        +----------------------------------------+
        <BLANKLINE>
        >>> major_pop = ld.major_population(LDPopulationName.NFE)
        >>> df.select(major_pop.col.alias("majorPopulation")).show(truncate=False)
        +---------------+
        |majorPopulation|
        +---------------+
        |{nfe, 0.9}     |
        |{nfe, 1.0}     |
        |{nfe, 0.5}     |
        |{eas, 0.5}     |
        |{nfe, 0.0}     |
        +---------------+
        <BLANKLINE>

        """
        initial_value = LDPopulation.from_components(ld_population_name=default_major_pop, relative_sample_size=0.0).col
        reducer: Callable[[Column, Column], Column]
        reducer = lambda x, y: self.find_larger_ld_population(LDPopulation(x), LDPopulation(y), default_major_pop).col

        expr = f.when(self.size > 0, f.reduce(self.col, initial_value, reducer)).otherwise(initial_value)
        return LDPopulation(expr)

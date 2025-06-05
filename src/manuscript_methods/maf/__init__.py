"""MAF related methods."""

from enum import Enum

from pyspark.sql import Column
from pyspark.sql import functions as f


def major_population_in_study(ld_col: Column, default_major_pop: str = "nfe") -> Column:
    """Extract the major population from the study ld population structure.

    Args:
        ld_col (Column): ld population structure field  array<struct<ldPopulation: string, relativeSampleSize: double>>
        default_major_pop (str, optional): population to use as default, when no population was reported. Defaults to "nfe".

    Returns:
        Column: ld_col struct

    """

    def reduce_pops(pop1: Column, pop2: Column) -> Column:
        """Reduce two populations based on relative sample size.

        This function takes 2 populations and report one of them based on following conditions:
        * Use pop with bigger relativeSampleSize
        * In case of a tie, the default_major_pop is preferred,
        * In case of tie and no default_major_pop in pop1 and pop2, use pop1.
        """
        return (
            f.when(pop1.getField("relativeSampleSize") > pop2.getField("relativeSampleSize"), pop1)
            .when(pop1.getField("relativeSampleSize") < pop2.getField("relativeSampleSize"), pop2)
            .when(
                (
                    (pop1.getField("relativeSampleSize") == pop2.getField("relativeSampleSize"))
                    & (pop1.getField("ldPopulation") == f.lit(default_major_pop))
                ),
                pop1,
            )
            .when(
                (
                    (pop1.getField("relativeSampleSize") == pop2.getField("relativeSampleSize"))
                    & (pop2.getField("ldPopulation") == f.lit(default_major_pop))
                ),
                pop2,
            )
            .otherwise(pop1)
        )

    fallback = f.struct(f.lit(default_major_pop).alias("ldPopulation"), f.lit(0.0).alias("relativeSampleSize"))

    return f.when(
        f.size(ld_col) > 0,
        f.reduce(
            ld_col,
            fallback,
            reduce_pops,
        ),
    ).otherwise(fallback)


def major_population_allele_freq(major_pop: Column, allele_freq: Column) -> Column:
    """Extract major population from variant.alleleFrequencies."""
    return f.filter(
        allele_freq,
        lambda freq: f.replace(freq.getField("populationName"), f.lit("_adj"), f.lit(""))
        == major_pop.getField("ldPopulation"),
    )


def maf(variant_freq: Column) -> Column:
    """Calculate Minor Allele Frequency from variant frequency."""
    return (
        f.when(
            ((f.size(variant_freq) == 1) & (variant_freq.getItem(0).getField("alleleFrequency") > 0.5)),
            f.lit(1.0) - variant_freq.getItem(0).getField("alleleFrequency"),
        )
        .when(
            ((f.size(variant_freq) == 1) & (variant_freq.getItem(0).getField("alleleFrequency") <= 0.5)),
            variant_freq.getItem(0).getField("alleleFrequency"),
        )
        .otherwise(None)
    )


class MAFDiscrepancies(str, Enum):
    """Enum representing MAF discrepancies."""

    MAJOR_ANCESTRY_MISSING_FROM_GNOMAD_AF = "MAJOR_ANCESTRY_MISSING_FROM_GNOMAD_AF"
    VARIANT_MISSING_FROM_GNOMAD_AF = "VARIANT_MISSING_FROM_GNOMAD_AF"


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

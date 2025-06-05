"""Dataset containing the information about the lead variant MAF."""

from enum import Enum

from gentropy.dataset.study_index import StudyIndex
from gentropy.dataset.study_locus import StudyLocus
from gentropy.dataset.variant_index import VariantIndex
from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from manuscript_methods.maf import maf, major_population_allele_freq, major_population_in_study
from manuscript_methods.pip import extract_pip_from_locus
from manuscript_methods.vep import vep_variant_effect


def create_maf_dataset(si: StudyIndex, cs: StudyLocus, vi: VariantIndex) -> DataFrame:
    """Create MAF dataset from StudyIndex, StudyLocus and VariantIndex.

    Args:
        si (StudyIndex): StudyIndex object
        cs (StudyLocus): StudyLocus object
        vi (VariantIndex): VariantIndex object

    Returns:
        DataFrame: MAF dataset

    """
    _cs = cs.df.select(
        f.col("studyId"),
        f.col("studyLocusId"),
        f.col("variantId"),
        f.col("beta"),
        f.col("zScore"),
        f.col("pValueMantissa"),
        f.col("pValueExponent"),
        f.col("standardError"),
        f.col("finemappingMethod"),
        f.col("studyType"),
        f.size("locus").alias("credibleSetSize"),
        f.col("isTransQtl"),
        extract_pip_from_locus(f.col("variantId"), f.col("locus")),
    )
    _si = si.df.select(
        f.col("studyId"),
        f.col("nSamples"),
        f.col("nControls"),
        f.col("nCases"),
        f.col("geneId"),  # for molqtl traits
        f.col("traitFromSourceMappedIds"),
        major_population_in_study(f.col("ldPopulationStructure"), "nfe").alias("majorPopulation"),
    )

    _vi = vi.df.select(
        f.col("variantId"),
        f.col("allelefrequencies"),
        vep_variant_effect(f.col("variantEffect")).alias("vepEffect"),
    )

    return (
        _cs.join(_si, how="left", on="studyId")
        .join(_vi, how="left", on="variantId")
        .select(
            "*",
            major_population_allele_freq(
                f.col("majorPopulation"),
                f.col("alleleFrequencies"),
            ).alias("majorPopulationAF"),
        )
        .select(
            "*",
            maf(f.col("majorPopulationAf")).alias("majorPopulationMAF"),
        )
    )

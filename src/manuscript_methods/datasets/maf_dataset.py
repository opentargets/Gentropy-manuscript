"""Dataset containing the information about the lead variant MAF."""

from enum import Enum

from gentropy.dataset.study_index import StudyIndex
from gentropy.dataset.study_locus import StudyLocus
from gentropy.dataset.variant_index import VariantIndex
from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from manuscript_methods.ld_populations import LDPopulationName, LDPopulationStructure
from manuscript_methods.maf import AlleleFrequencies
from manuscript_methods.pip import extract_pip_from_locus
from manuscript_methods.ve import SingleVariantEffectMethod, VariantEffect


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
        f.col("ldPopulationStructure"),
    )

    _vi = vi.df.select(
        f.col("variantId"),
        f.col("allelefrequencies"),
        f.col("variantEffect"),
    )

    _join = _cs.join(_si, how="left", on="studyId").join(_vi, how="left", on="variantId")

    major_ld_population = LDPopulationStructure().major_population(LDPopulationName.NFE)
    vep_effect = VariantEffect().filter_effect_by_method(SingleVariantEffectMethod.VEP)
    major_ld_maf = AlleleFrequencies().major_ld_population_maf(major_ld_population.ld_population)

    return _join.select(
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
        f.col("credibleSetSize"),
        f.col("isTransQtl"),
        f.col("pip"),
        f.col("nSamples"),
        f.col("nControls"),
        f.col("nCases"),
        f.col("geneId"),  # for molqtl traits
        f.col("traitFromSourceMappedIds"),
        major_ld_maf,
        major_ld_population.col,
        vep_effect.col,
    )

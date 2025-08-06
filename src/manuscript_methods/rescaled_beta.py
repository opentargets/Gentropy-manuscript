"""Rescaled statistics for trait analysis."""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as f

from manuscript_methods.study_statistics import TraitClassName


class RescaledStatistics:
    """Class for rescaling beta values based on the trait class."""

    name = "rescaledStatistics"
    """Name of the rescaled statistics."""
    schema = "struct<estimatedBeta: FLOAT, estimatedSE: FLOAT, varG: FLOAT, prev: FLOAT>"

    def __init__(self, col: Column | None = None):
        """Initialize RescaledBeta with an optional column.

        Args:
            col (f.Column, optional): Optional column to initialize the rescaled beta.

        """
        self.col = col.alias(self.name) if col is not None else f.col(self.name)

    @property
    def estimated_beta(self) -> Column:
        """Get the estimated beta value."""
        return self.col.getField("estimatedBeta").alias("estimatedBeta")

    @property
    def estimated_se(self) -> Column:
        """Get the estimated standard error."""
        return self.col.getField("estimatedSE").alias("estimatedSE")

    @property
    def var_g(self) -> Column:
        """Get the variance explained by the genotype."""
        return self.col.getField("varG").alias("varG")

    @property
    def prev(self) -> Column:
        """Get the prevalence of the trait."""
        return self.col.getField("prevalence").alias("prevalence")

    @classmethod
    def compute_beta_sign(cls, beta: Column) -> Column:
        """Determine the direction of effect based on beta value."""
        return f.when(beta < 0, f.lit(-1)).otherwise(f.lit(1)).alias("betaSign")

    @staticmethod
    def compute_direction_of_effect(beta: Column) -> Column:
        """Determine the direction of effect based on beta value."""
        return f.when(beta < 0, f.lit("-")).otherwise(f.lit("+")).alias("effectDirection")

    @classmethod
    def compute_z_score(cls, chi2_stat: Column, beta: Column) -> Column:
        """Calculate the z-score from the chi-squared statistic."""
        return cls.compute_beta_sign(beta) * f.sqrt(chi2_stat).alias("zScore")

    @classmethod
    def compute_effective_sample_size(cls, prev: Column, n_samples: Column) -> Column:
        """Calculate the effective sample size based on trait class."""
        return (prev * (1 - prev) * n_samples).alias("effectiveSampleSize")

    @classmethod
    def compute_var_g(cls, maf: Column) -> Column:
        """Calculate the variance explained by the additive genotype."""
        return (2 * maf * (1 - maf)).alias("varG")

    @classmethod
    def compute_prevalence(cls, n_cases: Column, n_samples: Column) -> Column:
        """Calculate the prevalence of the trait."""
        return (n_cases / n_samples).alias("prev")

    @classmethod
    def compute_se(
        cls, var_g: Column, n_samples: Column, trait_class: Column, prev: Column, var_phen: Column | None = None
    ) -> Column:
        """Calculate the standard error based on trait class.

        If `var_phen` is not provided, the method assumes that the phenotype was scaled to have a variance of 1 for quantitative traits.

        The definition of the standard errors is derived from the
        https://www.mv.helsinki.fi/home/mjxpirin/GWAS_course/material/GWAS3.pdf
        """
        var_phen = var_phen if isinstance(var_phen, Column) else f.lit(1.0)
        effective_n_samples = cls.compute_effective_sample_size(prev, n_samples)
        linear_se = f.sqrt(var_phen / (var_g * n_samples))
        logit_se = f.sqrt(1 / (var_g * effective_n_samples))
        return (
            f.when(trait_class == f.lit(TraitClassName.QUANTITATIVE), linear_se)
            .when(trait_class == f.lit(TraitClassName.BINARY), logit_se)
            .alias("se")
        )

    @classmethod
    def compute_minor_allele_rescaled_beta(cls, major_ancestry_af: Column, rescaled_beta: Column) -> Column:
        """Compute the minor allele rescaled beta based on the major ancestry allele frequency."""
        return (
            f.when(major_ancestry_af <= 0.5, rescaled_beta).otherwise(-rescaled_beta).alias("minorAlleleRescaledBeta")
        )

    @classmethod
    def compute(
        cls,
        chi2_stat: Column,
        trait_class: Column,
        beta: Column,
        maf: Column,
        af: Column,
        n_samples: Column,
        n_cases: Column,
    ) -> RescaledStatistics:
        beta_sign = cls.compute_direction_of_effect(beta)
        z_score = cls.compute_z_score(chi2_stat, beta)
        var_g = cls.compute_var_g(maf)
        prev = cls.compute_prevalence(n_cases, n_samples)
        se = cls.compute_se(var_g, n_samples, trait_class, prev)
        rescaled_beta = z_score * se
        minor_allele_rescaled_beta = cls.compute_minor_allele_rescaled_beta(af, rescaled_beta)

        return cls(
            f.struct(
                beta_sign.alias("directionOfEffect"),
                z_score.alias("zScore"),
                var_g.alias("varG"),
                prev.alias("prevalence"),
                se.alias("estimatedSE"),
                rescaled_beta.alias("estimatedBeta"),
                minor_allele_rescaled_beta.alias("minorAlleleEstimatedBeta"),
            )
        )

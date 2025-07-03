"""Transcript consequences."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from pyspark.sql import Column
from pyspark.sql import functions as f

from manuscript_methods.study_statistics import StudyStatistics, StudyType


class EffectType(str, Enum):
    """Effect type enumeration."""

    IN_GENE_EFFECT = "in-gene-effect"
    OUT_OF_GENE_EFFECT = "out-of-gene-effect"
    UNKNOWN = "unknown"


class TranscriptConsequence:
    name = "transcriptConsequence"

    def __init__(self, col: Column | None = None):
        self.col = col.alias(self.name) if col is not None else f.col(self.name)

    # attributes
    @property
    def variant_functional_consequence_ids(self) -> Column:
        """Variant functional consequence IDs."""
        return self.col.getField("variantFunctionalConsequenceIds")

    @property
    def amino_acid_change(self) -> Column:
        """Amino acid change."""
        return self.col.getField("aminoAcidChange")

    @property
    def uniprot_accession(self) -> Column:
        """UniProt accession."""
        return self.col.getField("uniprotAccessions")

    @property
    def is_ensembl_canonical(self) -> Column:
        """Is Ensembl canonical."""
        return self.col.getField("isEnsemblCanonical")

    @property
    def codons(self) -> Column:
        """Codons."""
        return self.col.getField("codons")

    @property
    def distance_from_footprint(self) -> Column:
        """Distance from footprint."""
        return self.col.getField("distanceFromFootprint")

    @property
    def distance_from_tss(self) -> Column:
        """Distance from TSS."""
        return self.col.getField("distanceFromTss")

    @property
    def appris(self) -> Column:
        """APPRIS."""
        return self.col.getField("appris")

    @property
    def mane_select(self) -> Column:
        """MANE Select."""
        return self.col.getField("maneSelect")

    @property
    def target_id(self) -> Column:
        """Target ID."""
        return self.col.getField("targetId")

    @property
    def impact(self) -> Column:
        """Impact."""
        return self.col.getField("impact")

    @property
    def loftee_prediction(self) -> Column:
        """LOFTEE prediction."""
        return self.col.getField("lofteePrediction")

    @property
    def sift_prediction(self) -> Column:
        """SIFT prediction."""
        return self.col.getField("siftPrediction")

    @property
    def polyphen_prediction(self) -> Column:
        """PolyPhen prediction."""
        return self.col.getField("polyphenPrediction")

    @property
    def consequence_score(self) -> Column:
        """Consequence score."""
        return self.col.getField("consequenceScore")

    @property
    def transcript_index(self) -> Column:
        """Transcript index."""
        return self.col.getField("transcriptIndex")

    @property
    def approved_symbol(self) -> Column:
        """Approved symbol."""
        return self.col.getField("approvedSymbol")

    @property
    def biotype(self) -> Column:
        """Biotype."""
        return self.col.getField("biotype")

    @property
    def transcript_id(self) -> Column:
        """Transcript ID."""
        return self.col.getField("transcriptId")

    def __eq__(self, other: TranscriptConsequence) -> Column:  # type: ignore
        """Equality operator."""
        return self.consequence_score == other.consequence_score

    def __ne__(self, other: TranscriptConsequence) -> Column:  # type: ignore
        """Inequality operator."""
        return self.consequence_score != other.consequence_score

    def __lt__(self, other: TranscriptConsequence) -> Column:
        """Less than operator."""
        return self.consequence_score < other.consequence_score

    def __le__(self, other: TranscriptConsequence) -> Column:
        """Less than or equal to operator."""
        return self.consequence_score <= other.consequence_score

    def __gt__(self, other: TranscriptConsequence) -> Column:
        """Greater than operator."""
        return self.consequence_score > other.consequence_score

    def __ge__(self, other: TranscriptConsequence) -> Column:
        """Greater than or equal to operator."""
        return self.consequence_score >= other.consequence_score

    def contains_molecular_trait(self, molecular_trait: Column) -> Column:
        """Check if the transcript consequence contains a specific molecular trait."""
        return self.target_id == molecular_trait


class TranscriptConsequences:
    """Transcript consequences."""

    name = "transcriptConsequences"

    def __init__(self, col: Column | None = None):
        self.col = col.alias(self.name) if col is not None else f.col(self.name)

    @staticmethod
    def find_more_severe_consequence(tc1: TranscriptConsequence, tc2: TranscriptConsequence) -> TranscriptConsequence:
        """Compare consequence severity."""
        return TranscriptConsequence(f.when(tc1.consequence_score > tc2.consequence_score, tc1.col).otherwise(tc2.col))

    @property
    def most_severe_consequence(self) -> TranscriptConsequence:
        """Most severe consequence."""
        reducer: Callable[[Column, Column], Column]
        reducer = lambda x, y: self.find_more_severe_consequence(TranscriptConsequence(x), TranscriptConsequence(y)).col
        expr = f.reduce(self.col, initialValue=self.col.getItem(0), merge=reducer)

        return TranscriptConsequence(expr)

    def find_molecular_trait_consequences(self, molecular_trait: Column) -> TranscriptConsequences:
        """Find the molecular trait consequence."""
        _filter: Callable[[Column], Column]
        _filter = lambda tc: TranscriptConsequence(tc).contains_molecular_trait(molecular_trait)
        expr = f.filter(self.col, _filter)
        return TranscriptConsequences(expr)

    def find_most_severe_molecular_trait_consequence(self, molecular_trait: Column) -> TranscriptConsequence:
        """Find the most severe molecular trait consequence."""
        expr = self.find_molecular_trait_consequences(molecular_trait)
        return expr.most_severe_consequence

    def contains_consequence_for_molecular_trait(self, molecular_trait: Column) -> Column:
        """Check if the transcript consequences contain a specific molecular trait."""
        return f.exists(self.col, lambda tc: TranscriptConsequence(tc).contains_molecular_trait(molecular_trait))


class LeadVariantConsequences:
    """Lead variant consequences."""

    name = "leadVariantConsequence"

    def __init__(self, col: Column | None = None):
        self.col = col.alias(self.name) if col is not None else f.col(self.name)

    @classmethod
    def compute(
        cls, transcript_consequences: TranscriptConsequences, study_statistics: StudyStatistics
    ) -> LeadVariantConsequences:
        """Compute lead variant consequences.

        For molecular traits, where the geneId from study exists in the transcript consequences,
        we will return the most severe consequence for the molecular trait and most severe consequence overall.

        If the study type is GWAS or molecular trait where we do not find the geneId in transcript consequences,
        we will return the most severe consequence only.
        """
        expr = f.when(
            (study_statistics.study_type != f.lit(StudyType.GWAS))
            & (transcript_consequences.contains_consequence_for_molecular_trait(study_statistics.molecular_trait)),
            f.struct(
                f.struct(
                    f.lit(EffectType.IN_GENE_EFFECT).alias("type"),
                    transcript_consequences.most_severe_consequence.col,
                ).alias("mostSevereConsequence"),
                f.struct(
                    f.lit(EffectType.OUT_OF_GENE_EFFECT).alias("type"),
                    transcript_consequences.find_most_severe_molecular_trait_consequence(
                        study_statistics.molecular_trait
                    ).col,
                ).alias("mostSevereConsequenceForMolecularTrait"),
            ),
        ).otherwise(
            f.struct(
                f.struct(
                    f.lit(EffectType.IN_GENE_EFFECT).alias("type"),
                    transcript_consequences.most_severe_consequence.col,
                ).alias("mostSevereConsequence"),
                f.struct(
                    f.lit(EffectType.UNKNOWN).alias("type"),
                    f.lit(None).alias(TranscriptConsequence.name),
                ).alias("mostSevereConsequenceForMolecularTrait"),
            ),
        )

        return cls(expr)

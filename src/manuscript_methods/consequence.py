"""Consequence categories and their classification based on Sequence Ontology (SO) terms."""

from enum import Enum

from pyspark.sql import functions as f
from pyspark.sql.column import Column


class IntergenicConsequence(str, Enum):
    """Enum for intergenic consequences."""

    UPSTREAM = "upstream_gene_variant"
    DOWNSTREAM = "downstream_gene_variant"
    INTERGENIC = "intergenic_variant"


class RegulatoryConsequence(str, Enum):
    """Enum for regulatory consequences."""

    TFBS_ABLATION = "TFBS_ablation"
    TFBS_AMPLIFICATION = "TFBS_amplification"
    TF_BINDING_SITE_VARIANT = "TF_binding_site_variant"
    REGULATORY_REGION_ABLATION = "regulatory_region_ablation"
    REGULATORY_REGION_AMPLIFICATION = "regulatory_region_amplification"
    REGULATORY_REGION_VARIANT = "regulatory_region_variant"


class IntragenicConsequence(str, Enum):
    """Enum for intragenic consequences."""

    SPLICE_DONOR_5TH_BASE_VARIANT = "splice_donor_5th_base_variant"
    SPLICE_REGION_VARIANT = "splice_region_variant"
    SPLICE_DONOR_REGION_VARIANT = "splice_donor_region_variant"
    SPLICE_POLYPYRIMIDINE_TRACT_VARIANT = "splice_polypyrimidine_tract_variant"
    INCOMPLETE_TERMINAL_CODON_VARIANT = "incomplete_terminal_codon_variant"
    START_RETAINED_VARIANT = "start_retained_variant"
    STOP_RETAINED_VARIANT = "stop_retained_variant"
    SYNONYMOUS_VARIANT = "synonymous_variant"
    CODING_SEQUENCE_VARIANT = "coding_sequence_variant"
    MATURE_MIRNA_VARIANT = "mature_miRNA_variant"
    FIVE_PRIME_UTR_VARIANT = "5_prime_UTR_variant"
    THREE_PRIME_UTR_VARIANT = "3_prime_UTR_variant"
    NON_CODING_TRANSCRIPT_EXON_VARIANT = "non_coding_transcript_exon_variant"
    INTRON_VARIANT = "intron_variant"
    NMD_TRANSCRIPT_VARIANT = "NMD_transcript_variant"
    NON_CODING_TRANSCRIPT_VARIANT = "non_coding_transcript_variant"
    CODING_TRANSCRIPT_VARIANT = "coding_transcript_variant"


class ProteinAlteringConsequence(str, Enum):
    """Enum for protein-altering consequences."""

    TRANSCRIPT_ABLATION = "transcript_ablation"
    SPLICE_ACCEPTOR_VARIANT = "splice_acceptor_variant"
    SPLICE_DONOR_VARIANT = "splice_donor_variant"
    STOP_GAINED = "stop_gained"
    FRAMESHIFT_VARIANT = "frameshift_variant"
    STOP_LOST = "stop_lost"
    START_LOST = "start_lost"
    TRANSCRIPT_AMPLIFICATION = "transcript_amplification"
    FEATURE_ELONGATION = "feature_elongation"
    FEATURE_TRUNCATION = "feature_truncation"
    INFRAME_INSERTION = "inframe_insertion"
    INFRAME_DELETION = "inframe_deletion"
    MISSENSE_VARIANT = "missense_variant"
    PROTEIN_ALTERING_VARIANT = "protein_altering_variant"


class ConsequenceCategory(str, Enum):
    """Enum for consequence categories."""

    INTERGENIC = "intergenic"
    REGULATORY = "regulatory"
    INTRAGENIC = "intragenic"
    PROTEIN_ALTERING = "protein_altering"

    @classmethod
    def classify_so_terms(cls, so_terms: Column) -> Column:
        """Classify SO terms into consequence categories."""
        return (
            f.when(so_terms.isin(*IntergenicConsequence.__members__.values()), f.lit(cls.INTERGENIC.value))
            .when(so_terms.isin(*RegulatoryConsequence.__members__.values()), f.lit(cls.REGULATORY.value))
            .when(so_terms.isin(*IntragenicConsequence.__members__.values()), f.lit(cls.INTRAGENIC.value))
            .when(so_terms.isin(*ProteinAlteringConsequence.__members__.values()), f.lit(cls.PROTEIN_ALTERING.value))
            .otherwise(None)
        )

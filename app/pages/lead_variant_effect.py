"""Lead variant effect page."""

from typing import Protocol

import pandas as pd
import plotly.express as px
import streamlit as st
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f

from manuscript_methods import break_string, group_statistics
from manuscript_methods.maf import MinorAlleleFrequency, MinorAlleleFrequencyClassification, maf_discrepancies

st.set_page_config(
    page_title="Lead Variant Effect",
    layout="wide",
    initial_sidebar_state="expanded",
)


class Module(Protocol):
    name: str
    description: str

    def display(self) -> None:
        """Display the module content."""
        self.heading()
        self.content()
        self.footer()

    def heading(self) -> None:
        """Return the heading for the module."""
        st.write(f"## {self.name}")
        st.write(self.description)

    def footer(self) -> None:
        """Return the footer for the module."""
        st.write("---")

    def content(self) -> None:
        """Return the content for the module."""
        raise NotImplementedError("Subclasses should implement this method.")


class LDStructureModule(Module):
    name = "LD Structure"
    description = "This module displays the LD structure of the lead variant effect dataset."

    def __init__(self, lead_variant_effect: DataFrame) -> None:
        """Initialize the LD structure module with the lead variant effect dataset."""
        self.lead_variant_effect = lead_variant_effect

    def content(self) -> None:
        """Display the LD structure content."""
        tab1, tab2 = st.tabs(["Counts per Major Ancestry and Study Type", "Counts per Major Ancestry"])
        with tab1:
            col1, col2 = st.columns([2, 3])
            col1.subheader("Overview")
            col1.write("This section provides an overview of the counts per major ancestry and study type")
            df = self._cs_counts_per_major_ancestry_and_study_type()
            chart = px.bar(df, x="ldPopulation", y="percentage", color="studyType")
            col1.plotly_chart(chart, use_container_width=True)
            col2.dataframe(df.set_index(["ldPopulation", "studyType"]).drop(columns=["percentage"]))
        with tab2:
            col1, col2 = st.columns([2, 3])
            col1.subheader("Overview")
            col1.write("This section provides an overview of the counts per major ancestry")
            df = self._cs_counts_per_major_ancestry()
            chart = px.bar(df, x="ldPopulation", y="percentage")
            col1.plotly_chart(chart, use_container_width=True)
            col2.dataframe(df.set_index("ldPopulation").drop(columns=["percentage"]))

    def _cs_counts_per_major_ancestry_and_study_type(self) -> pd.DataFrame:
        """Compute counts per major ancestry and study type."""
        data = self.lead_variant_effect.select("majorLDPopulation.ldPopulation", "studyStatistics.studyType")
        return (
            group_statistics(data, group_column=["ldPopulation", "studyType"])
            .select("ldPopulation", "studyType", f.format_number("count", 2).alias("count"), "%", "percentage")
            .toPandas()
        )

    def _cs_counts_per_major_ancestry(self) -> pd.DataFrame:
        """Compute counts per major ancestry."""
        data = self.lead_variant_effect.select("majorLDPopulation.ldPopulation")
        return (
            group_statistics(data, group_column=["ldPopulation"])
            .select("ldPopulation", f.format_number("count", 2).alias("count"), "%", "percentage")
            .toPandas()
        )


class EffectAlleleFrequencyModule(Module):
    name = "Effect Allele Frequency"
    description = "This module displays the effect allele frequency of the lead variant effect dataset."

    def __init__(self, lead_variant_effect: DataFrame) -> None:
        """Initialize the Effect Allele Frequency module with the lead variant effect dataset."""
        self.lead_variant_effect = lead_variant_effect

    def content(self) -> None:
        """Display the Effect Allele Frequency content."""
        tab1, tab2 = st.tabs(
            [
                "Distribution of GWAS lead variants Allele Frequency per Ancestry",
                "Distribution of lead variant Allele Frequency per Study Type",
            ]
        )
        df = self.lead_variant_effect.select(
            "majorLDPopulation.ldPopulation",
            "studyStatistics.studyType",
            "majorLdPopulationAf.alleleFrequency",
        )
        with tab1:
            col1, col2 = st.columns([2, 3])
            col1.subheader("Overview")
            col1.write("This section provides an overview of the distribution of allele frequency per ancestry")
            gwas_df = df.filter(f.col("studyStatistics.studyType") == "gwas")
            fig = px.box(gwas_df.toPandas(), x="ldPopulation", y="alleleFrequency")
            fig.update_traces(quartilemethod="exclusive")  # or "inclusive", or "linear" by default
            col1.plotly_chart(fig, use_container_width=True)
            group_data = group_statistics(df, group_column=["ldPopulation"])
            col2.dataframe(group_data.toPandas().set_index("ldPopulation").drop(columns=["percentage"]))

        with tab2:
            col1, col2 = st.columns([2, 3])
            col1.subheader("Overview")
            col1.write("This section provides an overview of the distribution of allele frequency per study type")

            fig = px.box(df.toPandas(), x="studyType", y="alleleFrequency")
            fig.update_traces(quartilemethod="exclusive")
            col1.plotly_chart(fig, use_container_width=True)
            group_data = group_statistics(df, group_column=["studyType"])
            col2.dataframe(group_data.toPandas().set_index("studyType").drop(columns=["percentage"]))


class MinorAlleleFrequencyModule(Module):
    name = "Minor Allele Frequency"
    description = "This module displays the minor allele frequency of the lead variant effect dataset."

    def __init__(self, lead_variant_effect: DataFrame) -> None:
        """Initialize the Minor Allele Frequency module with the lead variant effect dataset."""
        self.lead_variant_effect = lead_variant_effect

    def content(self) -> None:
        """Display the Effect Allele Frequency content."""
        tab1, tab2, tab3 = st.tabs(
            [
                "Distribution of GWAS lead variants Minor Allele Frequency per Ancestry",
                "Distribution of lead variant Minor Allele Frequency per Study Type",
                "Discrepancies in Minor Allele Frequency Computation",
            ]
        )
        df = self.lead_variant_effect.select(
            "majorLDPopulation.ldPopulation",
            "studyStatistics.studyType",
            f.col("majorLdPopulationMaf").getField("value").alias("minorAlleleFrequency"),
        )
        with tab1:
            col1, col2 = st.columns([2, 3])
            col1.subheader("Overview")
            col1.write("This section provides an overview of the distribution of minor allele frequency per ancestry")
            gwas_df = df.filter(f.col("studyStatistics.studyType") == "gwas")
            fig = px.box(gwas_df.toPandas(), x="ldPopulation", y="minorAlleleFrequency")
            fig.update_traces(quartilemethod="exclusive")  # or "inclusive", or "linear" by default
            col1.plotly_chart(fig, use_container_width=True)
            group_data = group_statistics(df, group_column=["ldPopulation"])
            col2.dataframe(group_data.toPandas().set_index("ldPopulation").drop(columns=["percentage"]))

        with tab2:
            col1, col2 = st.columns([2, 3])
            col1.subheader("Overview")
            col1.write("This section provides an overview of the distribution of minor allele frequency per study type")

            fig = px.box(df.toPandas(), x="studyType", y="minorAlleleFrequency")
            fig.update_traces(quartilemethod="exclusive")
            col1.plotly_chart(fig, use_container_width=True)
            group_data = group_statistics(df, group_column=["studyType"])
            col2.dataframe(group_data.toPandas().set_index("studyType").drop(columns=["percentage"]))

        with tab3:
            col1, col2 = st.columns([2, 3])
            col1.subheader("Overview")
            col1.write("This section provides an overview of the discrepancies in minor allele frequency computation")
            discrepancies = df.select(maf_discrepancies(f.col("minorAlleleFrequency")))
            group_discrepancies = group_statistics(discrepancies, group_column=["mafDiscrepancy"]).toPandas()
            group_discrepancies["mafDiscrepancy"] = group_discrepancies["mafDiscrepancy"].apply(break_string)
            chart = px.bar(
                group_discrepancies,
                x="mafDiscrepancy",
                y="percentage",
                text="count",
            )
            chart.update_traces(texttemplate="%{text:.2s}", textposition="outside")
            col1.plotly_chart(chart, use_container_width=True)
            col2.dataframe(group_discrepancies)


class PVEModule(Module):
    name = "PVE"
    description = "This module displays the PVE of the lead variant effect dataset."

    def __init__(self, lead_variant_effect: DataFrame) -> None:
        """Initialize the PVE module with the lead variant effect dataset."""
        self.lead_variant_effect = lead_variant_effect

    def content(self) -> None:
        """Display the PVE content."""
        st.write("Overview")
        df = self.lead_variant_effect.select("variantStatistics.ApproximatedVarianceExplained").toPandas()
        chart = px.histogram(data_frame=df, x="ApproximatedVarianceExplained", nbins=50, histnorm="probability density")
        st.plotly_chart(chart, use_container_width=True)


session = st.session_state.get("session")
datasets = st.session_state.get("datasets")

if datasets and session:
    if "lead_variant_effect" not in datasets:
        st.error("Lead Variant Effect dataset is not loaded. Please check the dataset loader.")
    lead_variant_effect = datasets.get("lead_variant_effect")
    if lead_variant_effect is None:
        st.error("Lead Variant Effect dataset is empty or not loaded correctly.")
    else:
        ld_module = LDStructureModule(lead_variant_effect)
        effect_allele_freq_module = EffectAlleleFrequencyModule(lead_variant_effect)
        minor_allele_freq_module = MinorAlleleFrequencyModule(lead_variant_effect)
        pve_module = PVEModule(lead_variant_effect)
        with st.sidebar:
            t1 = st.button("LD structure", on_click=ld_module.display)
            t2 = st.button("Effect allele frequency", on_click=effect_allele_freq_module.display)
            t3 = st.button("Minor allele frequency", on_click=minor_allele_freq_module.display)
            t4 = st.button("PVE", on_click=pve_module.display)


# PLAN

# TODO: Add a section about the PVE
# TODO: Add a section about the StudyStatistics
# TODO: Add a section about the Rescaled effect size
# TODO: Add a section about the VEP consequences (most severe)
# TODO: Add a section about the VEP consequences (egene)
# TODO: Add a section about the LocusStatistics
# TODO: Add a section about the variant types
# TODO: Add a section about the regressions
# TODO: Add a section about the E2G
# TODO: Add a section about the Epiraction

"""Streamlit app for reproducible results for the publication."""

import os
from pathlib import Path

import streamlit as st
from gentropy import Session, StudyIndex, StudyLocus, VariantIndex
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f

from manuscript_methods import group_statistics
from manuscript_methods.datasets import LeadVariantEffect
from manuscript_methods.resource import classify_by_datasource

ROOT_PATH = Path(os.getenv("DATASET_PATH", Path(__file__).parent / "data" / "25.06"))
LEAD_VARIANT_EFFECT_DATASET_PATH = ROOT_PATH / "lead_variant_effect"

st.set_page_config(page_title="Gentropy Manuscript", layout="wide", initial_sidebar_state="expanded")


def array_to_sorted_string(array: Column) -> Column:
    """Convert an array to a sorted string."""
    return f.array_join(f.sort_array(array), ", ")


class DatasetLoader:
    def __init__(self, output_path: Path, excluded_path: Path | None, lead_variant_effect_path: Path | None) -> None:
        """Class to load datasets for the Streamlit app."""
        self.output_path = output_path
        self.excluded_path = excluded_path

        # Output datasets (required)
        self.study_path = output_path / "study"
        self.credible_set_path = output_path / "credible_set"
        self.variant_path = output_path / "variant"
        self.disease_path = output_path / "disease"

        # Excluded datasets (optional)
        self.excluded_study_path = excluded_path / "study" if excluded_path else None
        self.excluded_credible_set_path = excluded_path / "credible_set" if excluded_path else None

        # Other datasets (optional)
        self.lead_variant_effect_path = lead_variant_effect_path if lead_variant_effect_path else None

    def session(self):
        """Create a session for the Spark application."""
        session = Session(extended_spark_conf={"spark.driver.memory": "50g"})

        if "session" not in st.session_state:
            st.session_state.session = session
        return session

    def load_datasets(self) -> dict[str, DataFrame]:
        """Load datasets from the specified paths."""
        with st.spinner("Loading datasets..."):
            session = self.session()
            datasets: dict[str, DataFrame] = {}

            if self.study_path.exists():
                study_index = StudyIndex.from_parquet(session, self.study_path.as_posix()).df
                datasets["study"] = study_index
            else:
                st.warning(f"Study dataset was not found under {self.study_path.as_posix()}.")

            if self.credible_set_path.exists():
                credible_set = StudyLocus.from_parquet(session, self.credible_set_path.as_posix()).df
                datasets["credible_set"] = credible_set
            else:
                st.warning(f"Credible set dataset was not found under {self.credible_set_path.as_posix()}.")

            if self.variant_path.exists():
                variants = VariantIndex.from_parquet(path=f"{self.output_path}/variant", session=session).df
                datasets["variant"] = variants
            else:
                st.warning(f"Variant dataset was not found under {self.variant_path.as_posix()}.")

            if self.disease_path.exists():
                disease = session.spark.read.parquet(self.disease_path.as_posix())
                datasets["disease"] = disease
            else:
                st.warning(f"Disease dataset was not found under {self.disease_path.as_posix()}.")

            if self.excluded_study_path and self.excluded_study_path.exists():
                excluded_studies = StudyIndex.from_parquet(session, self.excluded_study_path.as_posix()).df
                datasets["excluded_study"] = excluded_studies

            if self.excluded_credible_set_path and self.excluded_credible_set_path.exists():
                excluded_credible_set = StudyLocus.from_parquet(session, self.excluded_credible_set_path.as_posix()).df
                datasets["excluded_credible_set"] = excluded_credible_set

            if self.lead_variant_effect_path and self.lead_variant_effect_path.exists():
                lead_variant_effect = LeadVariantEffect.from_parquet(session, self.lead_variant_effect_path.as_posix())
                datasets["lead_variant_effect"] = lead_variant_effect.df
            if "datasets" not in st.session_state:
                st.session_state.datasets = datasets

            return datasets


def main():
    """Run streamlit app."""
    st.write("# Gentropy manuscript")

    ROOT_PATH = Path(os.getenv("DATASET_PATH", Path(__file__).parent.parent / "data" / "25.06"))
    output_path = ROOT_PATH / "output"

    add_excluded = st.toggle("Add excluded datasets", value=True)
    add_lve = st.toggle("add Lead Variant Effect dataset", value=True)
    st.info(
        "By default, the analysis is performed on only `valid` datasets, if the toggle is checked, then the analysis is performed on both `valid` and `excluded` datasets."
    )
    if add_excluded:
        excluded_path = ROOT_PATH / "excluded"
        st.info("Excluded datasets will be loaded.")
    else:
        excluded_path = None

    if add_lve:
        lead_variant_effect_path = ROOT_PATH / "lead_variant_effect"
        st.info("Lead Variant Effect dataset will be loaded.")
    else:
        lead_variant_effect_path = None

    if st.button("Load datasets"):
        st.session_state.excluded_path = excluded_path
        st.session_state.output_path = output_path
        st.session_state.lead_variant_effect_path = lead_variant_effect_path
        datasets = DatasetLoader(
            output_path=output_path, excluded_path=excluded_path, lead_variant_effect_path=lead_variant_effect_path
        ).load_datasets()
        st.write(lead_variant_effect_path)
        st.write("## Loaded datasets")
        for name, df in datasets.items():
            st.write(f"### {name}")
            st.write(f"{name} dataset loaded with {(df.count()):,} rows.")
            if name in ["study", "excluded_study"]:
                tab1, tab2, tab3, tab4 = st.tabs(["Overall", "StudyType", "DataSource", "QC"])
                with tab1:
                    st.write("### Overall statistics")
                    st.write("* rows: ", df.count())
                    st.write("* Unique studyId: ", df.select("studyId").distinct().count())
                with tab2:
                    st.dataframe(
                        group_statistics(df, group_column=["studyType"])
                        .select("studyType", f.format_number("count", 2).alias("count"), "%")
                        .toPandas()
                        .set_index("studyType")
                    )
                with tab3:
                    st.dataframe(
                        group_statistics(classify_by_datasource(df), ["dataSourceId"])
                        .select("dataSourceId", f.format_number("count", 2).alias("count"), "%")
                        .toPandas()
                        .set_index("dataSourceId")
                    )
                with tab4:
                    st.dataframe(
                        group_statistics(
                            df.withColumn("qualityControls", array_to_sorted_string(f.col("qualityControls"))),
                            ["qualityControls"],
                        )
                        .select("qualityControls", f.format_number("count", 2).alias("count"), "%")
                        .toPandas()
                        .set_index("qualityControls")
                    )

            if name in ["credible_set", "excluded_credible_set"]:
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overall", "StudyType", "DataSource", "FM-method", "QC"])
                with tab1:
                    st.write("#### Overall statistics")
                    st.write("* rows: ", df.count())
                    st.write("* Unique studyLocusId: ", df.select("studyLocusId").distinct().count())
                    st.write("* Unique studyId: ", df.select("studyId").distinct().count())

                with tab2:
                    st.dataframe(
                        group_statistics(df, group_column=["studyType"])
                        .select("studyType", f.format_number("count", 2).alias("count"), "%")
                        .toPandas()
                        .set_index("studyType")
                    )
                with tab3:
                    st.dataframe(
                        group_statistics(classify_by_datasource(df), group_column=["dataSourceId"])
                        .select("dataSourceId", f.format_number("count", 2).alias("count"), "%")
                        .toPandas()
                        .set_index("dataSourceId")
                    )
                with tab4:
                    st.dataframe(
                        group_statistics(df, ["finemappingMethod"])
                        .select("finemappingMethod", f.format_number("count", 2).alias("count"), "%")
                        .toPandas()
                        .set_index("finemappingMethod")
                    )
                with tab5:
                    st.dataframe(
                        group_statistics(
                            df.withColumn("qualityControls", array_to_sorted_string(f.col("qualityControls"))),
                            ["qualityControls"],
                        )
                        .select("qualityControls", f.format_number("count", 2).alias("count"), "%")
                        .toPandas()
                        .set_index("qualityControls")
                    )
            st.write("---")


if __name__ == "__main__":
    main()

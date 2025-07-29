"""Streamlit app for reproducible results for the publication."""

import os
from pathlib import Path

import streamlit as st
from gentropy import Session, StudyIndex, StudyLocus, VariantIndex
from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from manuscript_methods import group_statistics
from manuscript_methods.resource import classify_by_datasource

ROOT_PATH = Path(os.getenv("DATASET_PATH", "."))
st.set_page_config(
    page_title="Gentropy Manuscript",
    layout="wide",  # "centered" or "wide"
    initial_sidebar_state="expanded",  # or "collapsed"
)


class DatasetLoader:
    def __init__(self, output_path: str, excluded_path: str | None) -> None:
        """Class to load datasets for the Streamlit app."""
        self.output_path = output_path
        self.excluded_path = excluded_path

    def session(self):
        """Create a session for the Spark application."""
        session = Session(extended_spark_conf={"spark.driver.memory": "50g"})

        if "session" not in st.session_state:
            st.session_state.session = session
        return session

    def load_datasets(_self) -> dict[str, DataFrame]:
        """Load datasets from the specified paths."""
        with st.spinner("Loading datasets..."):
            session = _self.session()
            variants = None
            study_index = StudyIndex.from_parquet(path=f"{_self.output_path}/study", session=session).df.withColumn(
                "valid", f.lit(True)
            )
            credible_set = StudyLocus.from_parquet(
                path=f"{_self.output_path}/credible_set", session=session
            ).df.withColumn("valid", f.lit(True))
            variants = VariantIndex.from_parquet(path=f"{_self.output_path}/variant", session=session).df
            disease = session.spark.read.parquet(f"{_self.output_path}/disease").withColumn("valid", f.lit(True))
            if _self.excluded_path:
                study_index = study_index.unionByName(
                    StudyIndex.from_parquet(path=f"{_self.excluded_path}/study", session=session).df.withColumn(
                        "valid", f.lit(False)
                    )
                )
                credible_set = credible_set.unionByName(
                    StudyLocus.from_parquet(path=f"{_self.excluded_path}/credible_set", session=session).df.withColumn(
                        "valid", f.lit(False)
                    )
                )
            study_index = classify_by_datasource(study_index)
            credible_set = classify_by_datasource(credible_set)
            datasets = {"study": study_index, "credible_set": credible_set, "variant": variants, "disease": disease}
            if "datasets" not in st.session_state:
                st.session_state.datasets = datasets

            return datasets


def main():
    """Run streamlit app."""
    st.write("# Gentropy manuscript")
    st.sidebar.success("Select chapter to view")
    excluded_path = None
    output_path = (ROOT_PATH / "output").as_posix()

    add_excluded = st.toggle("Add excluded datasets", value=False)
    st.info(
        "By default, the analysis is performed on only `valid` datasets, if the toggle is checked, then the analysis is performed on both `valid` and `excluded` datasets."
    )
    if add_excluded:
        excluded_path = (ROOT_PATH / "excluded").as_posix()
        st.warning("Excluded datasets will be loaded.")
    else:
        excluded_path = None

    if st.button("Load datasets"):
        st.session_state.excluded_path = excluded_path
        st.session_state.output_path = output_path
        datasets = DatasetLoader(
            output_path=(ROOT_PATH / "output").as_posix(), excluded_path=excluded_path
        ).load_datasets()

        st.write("## Loaded datasets")
        for name, df in datasets.items():
            st.write(f"### {name}")
            st.write(f"{name} dataset loaded with {(df.count()):,} rows.")
            if name in ["study"]:
                tab1, tab2, tab3 = st.tabs(["Overall", "StudyType", "DataSource"])
                with tab1:
                    st.write("### Overall statistics")
                    st.write("* valid rows: ", df.filter(f.col("valid")).count())
                    st.write("* invalid rows: ", df.filter(~f.col("valid")).count())
                    st.write("* Unique studyId: ", df.select("studyId").distinct().count())
                with tab2:
                    st.dataframe(
                        group_statistics(df, group_column=["studyType", "valid"])
                        .select("studyType", "valid", f.format_number("count", 2).alias("count"), "%")
                        .toPandas()
                        .set_index("studyType")
                    )
                with tab3:
                    st.dataframe(
                        group_statistics(df, ["dataSourceId", "valid"])
                        .select("dataSourceId", "valid", f.format_number("count", 2).alias("count"), "%")
                        .toPandas()
                        .set_index("dataSourceId")
                    )

            if name in ["credible_set"]:
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overall", "StudyType", "DataSource", "FM-method", "Size"])
                with tab1:
                    st.write("### Overall statistics")
                    st.write("* valid rows: ", df.filter(f.col("valid")).count())
                    st.write("* invalid rows: ", df.filter(~f.col("valid")).count())
                    st.write("* Unique studyLocusId: ", df.select("studyLocusId").distinct().count())
                    st.write("* Unique studyId: ", df.select("studyId").distinct().count())

                with tab2:
                    st.dataframe(
                        group_statistics(df, group_column=["studyType", "valid"])
                        .select("studyType", "valid", f.format_number("count", 2).alias("count"), "%")
                        .toPandas()
                        .set_index("studyType")
                    )
                with tab3:
                    st.dataframe(
                        group_statistics(df, ["dataSourceId", "valid"])
                        .select("dataSourceId", "valid", f.format_number("count", 2).alias("count"), "%")
                        .toPandas()
                        .set_index("dataSourceId")
                    )
                with tab4:
                    st.dataframe(
                        group_statistics(df, ["finemappingMethod", "valid"])
                        .select("finemappingMethod", "valid", f.format_number("count", 2).alias("count"), "%")
                        .toPandas()
                        .set_index("finemappingMethod")
                    )
                with tab5:
                    st.dataframe(
                        group_statistics(df.select(f.size("locus").alias("size"), "valid"), ["size", "valid"])
                        .filter(f.col("percentage") > 0.1)
                        .select("size", "valid", f.format_number("count", 2).alias("count"), "%")
                        .toPandas()
                        .set_index("size")
                    )
                    largest_locus = (
                        df.select("studyId", "studyLocusId", f.size("locus").alias("size"))
                        .orderBy(f.desc("size"))
                        .first()
                    )
                    if largest_locus is None:
                        st.warning("No largest loci found.")
                    else:
                        largest_locus_id = largest_locus["studyLocusId"]
                        study_id = largest_locus["studyId"]
                        size = largest_locus["size"]
                        st.markdown(f"**Largest locus**: {largest_locus_id} from {study_id} with {size:,} variants.")
                    st.warning("Loci sizes with count < 0.1 are not shown.")
            st.write("---")


if __name__ == "__main__":
    main()

"""Streamlit app for reproducible results for the publication."""

import os

import pandas as pd
import streamlit as st
from gentropy import Session, StudyIndex, StudyLocus, VariantIndex
from pyspark.sql import functions as f

from manuscript_methods.resource import classify_by_resource, describe_resource

dataset_path = os.getenv("DATASET_PATH", "data")

st.header("Systematic ancestry-specific fine-mapping")


OUTPUTS_PATH = f"{dataset_path}/output"
EXCLUDED_PATH = f"{dataset_path}/excluded"


session = Session(extended_spark_conf={"spark.master.memory": "50g"})
spark = session.spark


def add_total_to_df(df: pd.DataFrame, count_col_name="count", variable_col_name="dataset") -> pd.DataFrame:
    """Add a total column to the DataFrame."""
    assert count_col_name in df.columns, f"DataFrame must contain a '{count_col_name}' column"
    assert variable_col_name in df.columns, f"DataFrame must contain a '{variable_col_name}' column"
    total = df["count"].sum()
    totals = pd.DataFrame({variable_col_name: ["total"], count_col_name: [total]})
    return pd.concat([df, totals])


valid_study_index = StudyIndex.from_parquet(path=f"{OUTPUTS_PATH}/study", session=session)
invalid_study_index = StudyIndex.from_parquet(path=f"{EXCLUDED_PATH}/study", session=session)
valid_credible_sets = StudyLocus.from_parquet(path=f"{OUTPUTS_PATH}/credible_set", session=session)
invalid_credible_sets = StudyLocus.from_parquet(path=f"{EXCLUDED_PATH}/credible_set", session=session)
valid_variants = VariantIndex.from_parquet(path=f"{OUTPUTS_PATH}/variant", session=session)


st.write("## Statistics")
valid_study_count = valid_study_index.df.count()
invalid_study_count = invalid_study_index.df.count()
valid_study_locus_count = valid_credible_sets.df.count()
invalid_study_locus_count = invalid_credible_sets.df.count()
variant_count = valid_variants.df.count()

df = spark.createDataFrame(
    data=[
        (f"{valid_study_count:,}", "valid_study_count"),
        (f"{invalid_study_count:,}", "invalid_study_count"),
        (f"{valid_study_locus_count:,}", "valid_study_locus_count"),
        (f"{invalid_study_locus_count:,}", "invalid_study_locus_count"),
        (f"{variant_count:,}", "variant_count"),
    ],
    schema=["count", "dataset"],
)
st.dataframe(df.toPandas().set_index("dataset"))


st.write("### Totals")

st.write(f"Total number of studies:    {valid_study_count + invalid_study_count:,}")
st.write(f"Total number of study loci: {valid_study_locus_count + invalid_study_locus_count:,}")

valid_studies_per_study_type = valid_study_index.df.groupBy("studyType").count().orderBy("studyType").toPandas()
invalid_studies_per_study_type = invalid_study_index.df.groupBy("studyType").count().orderBy("studyType").toPandas()

st.write("### Study types")
st.write("#### Valid studies")
st.dataframe(add_total_to_df(valid_studies_per_study_type, variable_col_name="studyType").set_index("studyType"))

st.write("#### Invalid studies")
st.dataframe(add_total_to_df(invalid_studies_per_study_type, variable_col_name="studyType").set_index("studyType"))


st.write("### Resource types")
st.write("#### Valid studies")
st.dataframe(describe_resource(valid_study_index.df).toPandas().set_index("resourceId"))

st.write("#### Invalid studies")
st.dataframe(describe_resource(invalid_study_index.df).toPandas().set_index("resourceId"))
st.write("#### Valid study loci")
st.dataframe(describe_resource(valid_credible_sets.df).toPandas().set_index("resourceId"))
st.write("#### Invalid study loci")
st.dataframe(describe_resource(invalid_credible_sets.df).toPandas().set_index("resourceId"))

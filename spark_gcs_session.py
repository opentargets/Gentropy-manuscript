"""
Create a SparkSession pre-configured for Google Cloud Storage (gs://) access.

This script only sets Spark/Hadoop configuration keys on the SparkSession.
It does NOT install the GCS connector jar; use `install_gcs_connector.py`
for that (or ensure the connector is already available on the classpath).
"""

from __future__ import annotations

import os
from typing import Optional

from pyspark.sql import SparkSession


def create_spark_session(
    *,
    app_name: str = "gentropy-manuscript",
    gs_project_id: Optional[str] = None,
    service_account_keyfile: Optional[str] = None,
) -> SparkSession:
    """
    Create (or get) a SparkSession with GCS settings.

    Args:
        app_name: Spark application name.
        gs_project_id: Optional GCP project ID for bucket listing/creation.
            If None, reads from env var GCS_PROJECT_ID, else defaults to "".
        service_account_keyfile: Path to a service-account JSON key file.
            If None, reads from env var GOOGLE_APPLICATION_CREDENTIALS,
            else defaults to "".
    """

    if gs_project_id is None:
        gs_project_id = os.getenv("GCS_PROJECT_ID", "")

    if service_account_keyfile is None:
        service_account_keyfile = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

    return (
        SparkSession.builder.appName(app_name)
        # The AbstractFileSystem for 'gs:' URIs
        .config(
            "spark.hadoop.fs.AbstractFileSystem.gs.impl",
            "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS",
        )
        # Hadoop FileSystem implementation for 'gs:' URIs (needed by many Spark/Hadoop code paths)
        .config(
            "spark.hadoop.fs.gs.impl",
            "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem",
        )
        # Optional. Google Cloud Project ID with access to GCS buckets.
        # Required only for list buckets and create bucket operations.
        .config("spark.hadoop.fs.gs.project.id", gs_project_id)
        # Whether to use a service account for GCS authorization.
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
        # The JSON keyfile of the service account used for GCS access.
        .config(
            "spark.hadoop.google.cloud.auth.service.account.json.keyfile",
            service_account_keyfile,
        )
        .getOrCreate()
    )


if __name__ == "__main__":
    spark = create_spark_session()
    print(f"Spark version: {spark.version}")
    for k in [
        "spark.hadoop.fs.AbstractFileSystem.gs.impl",
        "spark.hadoop.fs.gs.impl",
        "spark.hadoop.fs.gs.project.id",
        "spark.hadoop.google.cloud.auth.service.account.enable",
        "spark.hadoop.google.cloud.auth.service.account.json.keyfile",
    ]:
        print(f"{k}={spark.sparkContext.getConf().get(k, '')}")

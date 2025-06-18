"""Lead Variant Effect Dataset."""

import importlib.resources as pkg_resources
import json

from gentropy.dataset.dataset import Dataset
from pyspark.sql.types import StructType

from manuscript_methods import schemas


def parse_spark_schema(schema_json: str) -> StructType:
    """Parse Spark schema from JSON.

    Args:
        schema_json (str): JSON filename containing spark schema in the schemas package

    Returns:
        StructType: Spark schema

    """
    core_schema = json.loads(pkg_resources.read_text(schemas, schema_json, encoding="utf-8"))
    return StructType.fromJson(core_schema)


class LeadVariantEffect(Dataset):
    """Dataset for lead variant effect."""

    @classmethod
    def get_schema(cls) -> StructType:
        """Provide the schema for the LeadVariantEffect dataset.

        Returns:
            str: JSON string of the schema.

        """
        return parse_spark_schema("lead_variant_effect.json")

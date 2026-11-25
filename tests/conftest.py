import os

import pytest
from pyspark.sql import SparkSession

os.environ.setdefault("JAVA_HOME", "/opt/homebrew/opt/openjdk@21")


@pytest.fixture(scope="session")
def spark():
    s = (
        SparkSession.builder.master("local[1]")
        .appName("tests")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield s
    s.stop()
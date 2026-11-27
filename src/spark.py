"""One place to create the Spark session used by the pipeline scripts."""
import os

from pyspark.sql import SparkSession


def get_spark(app: str, driver_memory: str = "8g") -> SparkSession:
    """Local Spark session with the settings the notebooks used (Java 21 from Homebrew)."""
    os.environ.setdefault("JAVA_HOME", "/opt/homebrew/opt/openjdk@21")
    return (
        SparkSession.builder
        .appName(app)
        .config("spark.driver.memory", driver_memory)
        .getOrCreate()
    )

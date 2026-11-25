"""Time-based train/test split and test labels for the OTTO data."""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

WEEK_SECONDS = 7 * 24 * 3600


def time_split(events: DataFrame, test_days: int = 7) -> tuple[DataFrame, DataFrame, int]:
    """Split events by time.

    train: every event before the cutoff
    test: all events of sessions that start at or after the cutoff
    Returns (train, test, cutoff).
    """
    max_ts = events.agg(F.max("ts")).first()[0]
    cutoff = max_ts - test_days * 24 * 3600 + 1

    train = events.filter(F.col("ts") < cutoff)
    starts = events.groupBy("session").agg(F.min("ts").alias("start_ts"))
    test_sessions = starts.filter(F.col("start_ts") >= cutoff).select("session")
    test = events.join(test_sessions, "session")
    return train, test, cutoff


def make_test_labels(test: DataFrame, seed: int = 42) -> tuple[DataFrame, DataFrame]:
    """Cut each test session (2+ events) at a random point.

    input: events before the cut (what the model sees)
    labels: session, click (next clicked aid), carts (set), orders (set) after the cut
    """
    w = Window.partitionBy("session").orderBy("ts", "aid")
    numbered = test.withColumn("idx", F.row_number().over(w) - 1)

    cuts = (
        numbered.groupBy("session").agg(F.count("*").alias("n"))
        .filter(F.col("n") >= 2)
        .withColumn("cut", (F.floor(F.rand(seed=seed) * (F.col("n") - 1)) + 1).cast("int"))
    )

    joined = numbered.join(cuts, "session")
    test_input = joined.filter(F.col("idx") < F.col("cut")).select("session", "aid", "ts", "type")
    after = joined.filter(F.col("idx") >= F.col("cut"))

    next_click = after.filter(F.col("type") == 0).groupBy("session").agg(
        F.min_by("aid", "idx").alias("click")
    )
    carts = after.filter(F.col("type") == 1).groupBy("session").agg(
        F.collect_set("aid").alias("carts")
    )
    orders = after.filter(F.col("type") == 2).groupBy("session").agg(
        F.collect_set("aid").alias("orders")
    )

    labels = (
        cuts.select("session")
        .join(next_click, "session", "left")
        .join(carts, "session", "left")
        .join(orders, "session", "left")
    )
    return test_input, labels
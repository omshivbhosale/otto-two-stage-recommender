"""Raw OTTO sessions -> compact events, time-based split, test labels and the ranker week (Spark).

  python -m scripts.prepare_data

Writes to data/processed/:
  train_events               every event: session, aid, ts (seconds), type (0 click, 1 cart, 2 order)
  split_train, split_test    events before the final week / sessions that start in it
  test_input, test_labels    test sessions cut at a random point: what the model sees / what came after
  rank_train, rank_input, rank_labels
                             the same split one week earlier, for training the rankers
"""
import json
import time

from pyspark.sql import functions as F
from pyspark.sql import types as T

from scripts._common import DATA, REPORTS, ROOT
from src.split import make_test_labels, time_split
from src.spark import get_spark

RAW = ROOT / "data" / "raw" / "otto-recsys-train.jsonl"

RAW_SCHEMA = T.StructType([
    T.StructField("session", T.LongType()),
    T.StructField("events", T.ArrayType(T.StructType([
        T.StructField("aid", T.LongType()),
        T.StructField("ts", T.LongType()),
        T.StructField("type", T.StringType()),
    ]))),
])


def label_stats(labels) -> dict:
    return labels.agg(
        F.count("*").alias("sessions"),
        F.count("click").alias("with_click"),
        F.count("carts").alias("with_carts"),
        F.count("orders").alias("with_orders"),
    ).first().asDict()


def main() -> None:
    start = time.time()
    spark = get_spark("prepare-data")
    read = lambda name: spark.read.parquet(str(DATA / name))  # noqa: E731
    write = lambda df, name: df.write.mode("overwrite").parquet(str(DATA / name))  # noqa: E731

    # 1. raw JSON lines -> one row per event, small integer types
    type_map = F.create_map(F.lit("clicks"), F.lit(0), F.lit("carts"), F.lit(1), F.lit("orders"), F.lit(2))
    events = (
        spark.read.schema(RAW_SCHEMA).json(str(RAW))
        .select("session", F.explode("events").alias("e"))
        .select(
            F.col("session").cast("int"),
            F.col("e.aid").cast("int").alias("aid"),
            (F.col("e.ts") / 1000).cast("int").alias("ts"),  # milliseconds -> seconds
            type_map[F.col("e.type")].cast("byte").alias("type"),
        )
    )
    write(events, "train_events")
    ev = read("train_events")
    print(f"events: {ev.count():,} | {(time.time() - start) / 60:.1f} min", flush=True)

    # 2. test week: sessions that start in the final 7 days
    train, test, cutoff = time_split(ev)
    write(train, "split_train")
    write(test, "split_test")
    test_input, test_labels = make_test_labels(read("split_test"), seed=42)
    write(test_input, "test_input")
    write(test_labels, "test_labels")

    # 3. ranker week: the same procedure one week earlier, using only data before the test week
    rank_train, rank_week, rank_cutoff = time_split(read("split_train"))
    write(rank_train, "rank_train")
    rank_input, rank_labels = make_test_labels(rank_week, seed=7)
    write(rank_input, "rank_input")
    write(rank_labels, "rank_labels")

    # leakage checks: no training event at or after its cutoff
    for name, cut in [("split_train", cutoff), ("rank_train", rank_cutoff)]:
        last = read(name).agg(F.max("ts")).first()[0]
        assert last < cut, f"{name} reaches past its cutoff"

    summary = {
        "events": read("train_events").count(),
        "test_cutoff": cutoff,
        "rank_cutoff": rank_cutoff,
        "test_week": label_stats(read("test_labels")),
        "rank_week": label_stats(read("rank_labels")),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"done | {(time.time() - start) / 60:.1f} min")
    spark.stop()


if __name__ == "__main__":
    main()

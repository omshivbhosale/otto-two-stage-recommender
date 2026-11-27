"""Build candidates, ranker features and labels for one week (Spark).

  python -m scripts.build_features --week rank --version v2    # ranker week: weeks 1-2 -> week 3
  python -m scripts.build_features --week test --version v2    # test week:   weeks 1-3 -> week 4

v1 sources: session history, bought-together, popular items, GRU.
v2 adds viewed-together partners of the shopper's history items (click
co-visitation over the 14 days before the week's cutoff).
Everything is built from events before the week's cutoff.
"""
import argparse
import time

from pyspark.sql import functions as F

from scripts._common import DATA, FEATURE_DIRS
from src.baselines import build_covisit_clicks
from src.features import add_features, add_labels, build_candidates
from src.spark import get_spark

WEEK_SECONDS = 7 * 24 * 3600

WEEKS = {
    "rank": {"train": "rank_train", "input": "rank_input", "labels": "rank_labels",
             "gru": "rank_gru_candidates", "weeks_before_test": 1},
    "test": {"train": "split_train", "input": "test_input", "labels": "test_labels",
             "gru": "test_gru_candidates", "weeks_before_test": 0},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--week", choices=WEEKS, required=True)
    parser.add_argument("--version", choices=["v1", "v2"], default="v2")
    args = parser.parse_args()
    week = WEEKS[args.week]
    start = time.time()

    spark = get_spark(f"build-features-{args.week}-{args.version}")
    read = lambda name: spark.read.parquet(str(DATA / name))  # noqa: E731

    max_ts = read("train_events").agg(F.max("ts")).first()[0]
    test_cutoff = max_ts - WEEK_SECONDS + 1
    cut = test_cutoff - week["weeks_before_test"] * WEEK_SECONDS

    train = read(week["train"])
    inp, labels, gru = read(week["input"]), read(week["labels"]), read(week["gru"])
    train_max = train.agg(F.max("ts")).first()[0]
    assert train_max < cut, f"training events reach past the cutoff ({train_max} >= {cut})"
    print(f"week: {args.week} | version: {args.version} | cutoff: {cut} | last training event: {train_max}")

    clicks_cv = None
    if args.version == "v2":
        # viewed-together table: written once, then reused (also needed for serving)
        cv_path = DATA / f"{args.week}_covisit_clicks"
        build_covisit_clicks(train, cut).write.mode("overwrite").parquet(str(cv_path))
        clicks_cv = spark.read.parquet(str(cv_path))
        print(f"viewed-together pairs: {clicks_cv.count():,} | {(time.time() - start) / 60:.1f} min", flush=True)

    cands = build_candidates(inp, train, cut, gru=gru, clicks_cv=clicks_cv)
    data = add_labels(add_features(cands, inp, train, cut), labels)

    out = DATA / FEATURE_DIRS[(args.version, args.week)]
    if args.week == "test":
        # scored later in 20 chunks so it fits in memory
        data.withColumn("bucket", F.col("session") % 20).write.mode("overwrite").partitionBy("bucket").parquet(str(out))
    else:
        data.write.mode("overwrite").parquet(str(out))

    df = spark.read.parquet(str(out))
    stats = df.agg(
        F.count("*").alias("rows"),
        F.countDistinct("session").alias("sessions"),
        F.avg("y_click").alias("click_rate"),
    ).first()
    print(f"rows: {stats.rows:,} | sessions: {stats.sessions:,} | per session: {stats.rows / stats.sessions:.1f}"
          f" | click rate: {stats.click_rate:.4f}")
    print(f"saved {out.name} | {(time.time() - start) / 60:.1f} min")
    spark.stop()


if __name__ == "__main__":
    main()

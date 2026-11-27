"""Candidate generation, ranker features, and labels."""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.baselines import build_covisit_buys, popular_items

COUNT_COLS = ["n_clicks", "n_carts", "n_orders", "item_clicks", "item_carts", "item_orders"]


def build_candidates(
    inp: DataFrame,
    train_events: DataFrame,
    cut: int,
    gru: DataFrame | None = None,
    clicks_cv: DataFrame | None = None,
) -> DataFrame:
    """One row per (session, aid) candidate, with a score column per source.

    Sources: session history (hist_rank), bought-together (cov_score, cov_rank),
    popular items (pop_rank), optionally GRU (gru_rank, gru_sim) and optionally
    viewed-together partners of the history items (clk_score, clk_rank), from
    a table built by src.baselines.build_covisit_clicks.
    """
    spark = inp.sparkSession
    hist = (
        inp.groupBy("session", "aid").agg(F.max("ts").alias("last_ts"))
        .withColumn("hist_rank", F.row_number().over(
            Window.partitionBy("session").orderBy(F.desc("last_ts"), "aid")))
        .filter("hist_rank <= 20")
    )

    cv = build_covisit_buys(train_events)
    cov = (
        hist.join(cv, hist.aid == cv.aid_x)
        .groupBy("session", F.col("aid_y").alias("aid"))
        .agg(F.sum("count").alias("cov_score"))
        .withColumn("cov_rank", F.row_number().over(
            Window.partitionBy("session").orderBy(F.desc("cov_score"), "aid")))
        .filter("cov_rank <= 20")
    )

    top = popular_items(train_events, cut)
    pop = spark.createDataFrame([(a, r + 1) for r, a in enumerate(top)], "aid int, pop_rank int")
    pop_c = inp.select("session").distinct().crossJoin(pop)

    cands = (
        hist.select("session", "aid", "hist_rank")
        .join(cov.select("session", "aid", "cov_score", "cov_rank"), ["session", "aid"], "full")
        .join(pop_c, ["session", "aid"], "full")
    )
    if clicks_cv is not None:
        clk = (
            hist.join(clicks_cv, hist.aid == clicks_cv.aid_x)
            .groupBy("session", F.col("aid_y").alias("aid"))
            .agg(F.sum("count").alias("clk_score"))
            .withColumn("clk_rank", F.row_number().over(
                Window.partitionBy("session").orderBy(F.desc("clk_score"), "aid")))
            .filter("clk_rank <= 20")
        )
        cands = cands.join(clk.select("session", "aid", "clk_score", "clk_rank"), ["session", "aid"], "full")
    if gru is not None:
        cands = cands.join(gru.select("session", "aid", "gru_rank", "gru_sim"), ["session", "aid"], "full")
    return cands


def add_features(cands: DataFrame, inp: DataFrame, train_events: DataFrame, cut: int) -> DataFrame:
    """Add shopper-item, session, and item-popularity features."""
    inter = inp.groupBy("session", "aid").agg(
        F.sum((F.col("type") == 0).cast("int")).alias("n_clicks"),
        F.sum((F.col("type") == 1).cast("int")).alias("n_carts"),
        F.sum((F.col("type") == 2).cast("int")).alias("n_orders"),
        F.max("ts").alias("item_last_ts"),
    )
    sess = inp.groupBy("session").agg(
        F.count("*").alias("sess_len"),
        F.countDistinct("aid").alias("sess_items"),
        F.max("ts").alias("sess_end"),
    )
    week = train_events.filter(F.col("ts") >= cut - 7 * 24 * 3600)
    item_pop = week.groupBy("aid").agg(
        F.sum((F.col("type") == 0).cast("int")).alias("item_clicks"),
        F.sum((F.col("type") == 1).cast("int")).alias("item_carts"),
        F.sum((F.col("type") == 2).cast("int")).alias("item_orders"),
    )
    return (
        cands.join(inter, ["session", "aid"], "left")
        .join(sess, "session", "left")
        .join(item_pop, "aid", "left")
        .withColumn("secs_since", F.col("sess_end") - F.col("item_last_ts"))
        .drop("item_last_ts", "sess_end")
        .fillna(0, subset=COUNT_COLS)
    )


def add_labels(df: DataFrame, labels: DataFrame) -> DataFrame:
    """Add y_click, y_cart, y_order (1 if the candidate is a right answer)."""
    return (
        df.join(labels, "session", "left")
        .withColumn("y_click", (F.col("aid") == F.col("click")).cast("int"))
        .withColumn("y_cart", F.coalesce(F.array_contains("carts", F.col("aid")), F.lit(False)).cast("int"))
        .withColumn("y_order", F.coalesce(F.array_contains("orders", F.col("aid")), F.lit(False)).cast("int"))
        .fillna(0, subset=["y_click"])
        .drop("click", "carts", "orders")
    )
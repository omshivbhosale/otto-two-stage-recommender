"""Baseline recommenders: popular items, session history, bought-together."""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

def _empty():
    return F.array().cast("array<int>")


def popular_items(train: DataFrame, cutoff: int, days: int = 7, k: int = 20) -> list[int]:
    """The k items most carted or ordered in the last `days` before the cutoff."""
    start = cutoff - days * 24 * 3600
    rows = (
        train.filter((F.col("ts") >= start) & (F.col("type") >= 1))
        .groupBy("aid").count()
        .orderBy(F.desc("count"), "aid")
        .limit(k)
        .collect()
    )
    return [r["aid"] for r in rows]


def build_covisit_buys(train: DataFrame, max_per_session: int = 30, top_k: int = 20) -> DataFrame:
    """For each item, its top_k partners carted or ordered in the same session.

    Returns columns aid_x, aid_y, count, rank.
    """
    buys = (
        train.filter(F.col("type") >= 1)
        .withColumn("rn", F.row_number().over(
            Window.partitionBy("session").orderBy(F.desc("ts"))
        ))
        .filter(F.col("rn") <= max_per_session)
        .select("session", "aid")
        .dropDuplicates()
    )
    a, b = buys.alias("a"), buys.alias("b")
    pairs = a.join(
        b, (F.col("a.session") == F.col("b.session")) & (F.col("a.aid") != F.col("b.aid"))
    ).select(F.col("a.aid").alias("aid_x"), F.col("b.aid").alias("aid_y"))

    return (
        pairs.groupBy("aid_x", "aid_y").count()
        .withColumn("rank", F.row_number().over(
            Window.partitionBy("aid_x").orderBy(F.desc("count"), "aid_y")
        ))
        .filter(F.col("rank") <= top_k)
    )


def build_covisit_clicks(
    train: DataFrame,
    cutoff: int,
    days: int = 14,
    max_per_session: int = 30,
    window_secs: int = 24 * 3600,
    top_k: int = 20,
) -> DataFrame:
    """For each item, its top_k partners viewed in the same session ("viewed together").

    Uses events from the `days` before the cutoff (any type). Each session contributes
    its `max_per_session` most recently touched items; two items form a pair when the
    session touched them within `window_secs` of each other. A pair counts once per session.

    Returns columns aid_x, aid_y, count, rank (same shape as build_covisit_buys).
    """
    start = cutoff - days * 24 * 3600
    items = (
        train.filter(F.col("ts") >= start)
        .groupBy("session", "aid").agg(F.max("ts").alias("ts"))
        .withColumn("rn", F.row_number().over(
            Window.partitionBy("session").orderBy(F.desc("ts"), "aid")
        ))
        .filter(F.col("rn") <= max_per_session)
        .select("session", "aid", "ts")
    )
    a, b = items.alias("a"), items.alias("b")
    pairs = a.join(
        b,
        (F.col("a.session") == F.col("b.session"))
        & (F.col("a.aid") != F.col("b.aid"))
        & (F.abs(F.col("a.ts") - F.col("b.ts")) <= window_secs),
    ).select(F.col("a.aid").alias("aid_x"), F.col("b.aid").alias("aid_y"))

    return (
        pairs.groupBy("aid_x", "aid_y").count()
        .withColumn("rank", F.row_number().over(
            Window.partitionBy("aid_x").orderBy(F.desc("count"), "aid_y")
        ))
        .filter(F.col("rank") <= top_k)
    )


def _ranked_list(df: DataFrame, order_col: str, name: str, k: int) -> DataFrame:
    """Turn (session, aid, score) rows into one array per session, best first."""
    return (
        df.withColumn("r", F.row_number().over(
            Window.partitionBy("session").orderBy(F.desc(order_col), "aid")
        ))
        .filter(F.col("r") <= k)
        .groupBy("session")
        .agg(F.sort_array(F.collect_list(F.struct("r", "aid"))).alias("s"))
        .select("session", F.transform("s", lambda x: x["aid"]).alias(name))
    )


def recommend(
    test_input: DataFrame,
    sessions: DataFrame,
    popular: list[int],
    covisit: DataFrame | None = None,
    use_history: bool = True,
    k: int = 20,
) -> DataFrame:
    """Build k recommendations per session.

    Order: session history (most recent first), then bought-together partners,
    then popular items to fill the list. Returns columns session, items.
    """
    hist = test_input.groupBy("session", "aid").agg(F.max("ts").alias("last_ts"))
    base = sessions.select("session")

    parts = []
    if use_history:
        base = base.join(_ranked_list(hist, "last_ts", "hist", k), "session", "left")
        parts.append(F.coalesce("hist", _empty()))

    if covisit is not None:
        cand = (
            hist.join(covisit, hist.aid == covisit.aid_x)
            .groupBy("session", F.col("aid_y").alias("aid"))
            .agg(F.sum("count").alias("w"))
            .join(hist.select("session", "aid"), ["session", "aid"], "left_anti")
        )
        base = base.join(_ranked_list(cand, "w", "covisit", k), "session", "left")
        parts.append(F.coalesce("covisit", _empty()))
    parts.append(F.array(*[F.lit(a) for a in popular]).cast("array<int>"))
    return base.select(
        "session", F.slice(F.array_distinct(F.concat(*parts)), 1, k).alias("items")
    )
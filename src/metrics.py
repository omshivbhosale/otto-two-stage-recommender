"""Evaluation metrics for the OTTO recommender."""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

WEIGHTS = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}


def recall_at_20(preds: DataFrame, labels: DataFrame) -> dict:
    """Score predictions against held-out labels, OTTO style.

    preds: columns session, items (array of up to 20 aids)
    labels: columns session, click (int), carts (array), orders (array)
    """
    empty = F.array().cast("array<int>")
    df = labels.join(preds, "session", "left").withColumn(
        "items", F.coalesce(F.col("items"), empty)
    )

    clicks = (
        df.filter(F.col("click").isNotNull())
        .select(F.array_contains("items", F.col("click")).cast("int").alias("hit"))
        .agg(F.avg("hit"))
        .first()[0]
    )

    def set_recall(col: str) -> float | None:
        totals = (
            df.filter(F.col(col).isNotNull())
            .select(
                F.size(F.array_intersect(col, "items")).alias("hits"),
                F.least(F.size(col), F.lit(20)).alias("denom"),
            )
            .agg(F.sum("hits"), F.sum("denom"))
            .first()
        )
        if not totals[1]:
            return None
        return totals[0] / totals[1]

    result = {"clicks": clicks, "carts": set_recall("carts"), "orders": set_recall("orders")}
    if any(v is None for v in result.values()):
        result["score"] = None
    else:
        result["score"] = sum(WEIGHTS[k] * result[k] for k in WEIGHTS)
    return result
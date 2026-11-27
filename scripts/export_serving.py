"""Export the lookup tables the API needs, built from the same data as the test week (Spark).

  python -m scripts.export_serving

Everything is built from events before the test-week cutoff, exactly like the
test-week features in scripts.build_features:

  artifacts/covisit_buys.npz    bought-together partners per item (top 20)
  artifacts/covisit_clicks.npz  viewed-together partners per item (top 20)
  artifacts/item_pop.npy        clicks, carts, orders per item in the last 7 days
  artifacts/popular.npy         the 20 most carted/ordered items in the last 7 days

Co-visitation tables are stored as CSR arrays: the partners of item a are
partners[offsets[a]:offsets[a + 1]], with matching counts.
Run after `scripts.build_features --week test --version v2`, which writes the
viewed-together table this script reuses.
"""
import json
import time

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from scripts._common import ARTIFACTS, DATA, N_ITEMS, REPORTS
from src.baselines import build_covisit_buys, popular_items
from src.spark import get_spark

WEEK_SECONDS = 7 * 24 * 3600


def to_csr(pairs: pd.DataFrame) -> dict[str, np.ndarray]:
    """(aid_x, aid_y, count, rank) rows -> CSR arrays, partners in rank order."""
    pairs = pairs.sort_values(["aid_x", "rank"], kind="stable")
    sizes = np.bincount(pairs["aid_x"].to_numpy(), minlength=N_ITEMS)
    offsets = np.zeros(N_ITEMS + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    return {
        "offsets": offsets,
        "partners": pairs["aid_y"].to_numpy().astype(np.int32),
        "counts": pairs["count"].to_numpy().astype(np.int32),
    }


def main() -> None:
    start = time.time()
    spark = get_spark("export-serving")
    read = lambda name: spark.read.parquet(str(DATA / name))  # noqa: E731

    train = read("split_train")
    cut = read("train_events").agg(F.max("ts")).first()[0] - WEEK_SECONDS + 1
    train_max = train.agg(F.max("ts")).first()[0]
    assert train_max < cut, f"training events reach past the cutoff ({train_max} >= {cut})"
    ARTIFACTS.mkdir(exist_ok=True)

    # bought-together: same call as the test-week features
    buys_path = DATA / "test_covisit_buys"
    build_covisit_buys(train).write.mode("overwrite").parquet(str(buys_path))
    buys = pd.read_parquet(buys_path)
    np.savez(ARTIFACTS / "covisit_buys.npz", **to_csr(buys))
    print(f"bought-together pairs: {len(buys):,} | {(time.time() - start) / 60:.1f} min", flush=True)

    # viewed-together: the table scripts.build_features wrote for the test week
    clicks = pd.read_parquet(DATA / "test_covisit_clicks")
    np.savez(ARTIFACTS / "covisit_clicks.npz", **to_csr(clicks))
    print(f"viewed-together pairs: {len(clicks):,}", flush=True)

    # item activity in the 7 days before the cutoff (same window as add_features)
    pop_path = DATA / "test_item_pop"
    (
        train.filter(F.col("ts") >= cut - WEEK_SECONDS)
        .groupBy("aid").agg(
            F.sum((F.col("type") == 0).cast("int")).alias("item_clicks"),
            F.sum((F.col("type") == 1).cast("int")).alias("item_carts"),
            F.sum((F.col("type") == 2).cast("int")).alias("item_orders"),
        )
        .write.mode("overwrite").parquet(str(pop_path))
    )
    pop = pd.read_parquet(pop_path)
    item_pop = np.zeros((N_ITEMS, 3), dtype=np.int32)
    item_pop[pop["aid"].to_numpy()] = pop[["item_clicks", "item_carts", "item_orders"]].to_numpy()
    np.save(ARTIFACTS / "item_pop.npy", item_pop)

    popular = np.array(popular_items(train, cut), dtype=np.int32)
    np.save(ARTIFACTS / "popular.npy", popular)
    spark.stop()

    summary = {
        "cutoff": int(cut),
        "bought_together_pairs": int(len(buys)),
        "viewed_together_pairs": int(len(clicks)),
        "items_active_last_7_days": int(len(pop)),
        "popular": popular.tolist(),
    }
    (REPORTS / "serving_artifacts.json").write_text(json.dumps(summary, indent=2))
    print(f"saved artifacts/ and reports/serving_artifacts.json | {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()

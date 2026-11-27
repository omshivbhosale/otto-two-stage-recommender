"""Score the simple baselines on the test week (Spark).

  python -m scripts.baselines

Popular items only, session history + popular, and history + bought-together + popular.
Writes reports/baselines_test_week.json.
"""
import json
import time

from pyspark.sql import functions as F

from scripts._common import DATA, REPORTS
from src.baselines import build_covisit_buys, popular_items, recommend
from src.metrics import recall_at_20
from src.spark import get_spark


def main() -> None:
    start = time.time()
    spark = get_spark("baselines")
    train = spark.read.parquet(str(DATA / "split_train"))
    test_input = spark.read.parquet(str(DATA / "test_input"))
    labels = spark.read.parquet(str(DATA / "test_labels"))
    cutoff = spark.read.parquet(str(DATA / "train_events")).agg(F.max("ts")).first()[0] - 7 * 24 * 3600 + 1

    popular = popular_items(train, cutoff)
    covisit = build_covisit_buys(train)
    sessions = labels.select("session")

    systems = {
        "popular items only": recommend(test_input, sessions, popular, use_history=False),
        "session history + popular": recommend(test_input, sessions, popular),
        "history + bought-together + popular": recommend(test_input, sessions, popular, covisit=covisit),
    }
    results = {}
    for name, preds in systems.items():
        results[name] = recall_at_20(preds, labels)
        r = results[name]
        print(f"{name:38s} clicks {r['clicks']:.4f} | carts {r['carts']:.4f} | "
              f"orders {r['orders']:.4f} | score {r['score']:.4f}", flush=True)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "baselines_test_week.json").write_text(json.dumps({"week": "test", "results": results}, indent=2))
    print(f"saved reports/baselines_test_week.json | {(time.time() - start) / 60:.1f} min")
    spark.stop()


if __name__ == "__main__":
    main()

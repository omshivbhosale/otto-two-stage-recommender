import pytest

from src.metrics import recall_at_20

LABEL_SCHEMA = "session int, click int, carts array<int>, orders array<int>"
PRED_SCHEMA = "session int, items array<int>"


@pytest.fixture
def labels(spark):
    return spark.createDataFrame(
        [(1, 10, [20, 21], [30]), (2, 11, None, [31, 32])], LABEL_SCHEMA
    )


def test_hand_example(spark, labels):
    preds = spark.createDataFrame([(1, [10, 20, 99]), (2, [5, 31])], PRED_SCHEMA)
    r = recall_at_20(preds, labels)
    assert r["clicks"] == pytest.approx(0.5)
    assert r["carts"] == pytest.approx(0.5)
    assert r["orders"] == pytest.approx(1 / 3)
    assert r["score"] == pytest.approx(0.4)


def test_perfect_predictions_score_one(spark, labels):
    preds = spark.createDataFrame([(1, [10, 20, 21, 30]), (2, [11, 31, 32])], PRED_SCHEMA)
    r = recall_at_20(preds, labels)
    assert r == pytest.approx({"clicks": 1.0, "carts": 1.0, "orders": 1.0, "score": 1.0})


def test_missing_session_counts_as_miss(spark, labels):
    preds = spark.createDataFrame([(1, [10])], PRED_SCHEMA)
    r = recall_at_20(preds, labels)
    assert r["clicks"] == pytest.approx(0.5)
    assert r["orders"] == pytest.approx(0.0)


def test_denominator_capped_at_20(spark):
    many = list(range(100, 125))
    labels = spark.createDataFrame([(1, None, None, many)], LABEL_SCHEMA)
    preds = spark.createDataFrame([(1, many[:20])], PRED_SCHEMA)
    r = recall_at_20(preds, labels)
    assert r["orders"] == pytest.approx(1.0)
import math

import pandas as pd
import pytest

from src.ranking import blend_scores, otto_recall, rank_within_session, top_k_lists


@pytest.fixture
def cands():
    # two sessions, scores from two rankers
    return pd.DataFrame({
        "session": [1, 1, 1, 2, 2],
        "aid":     [10, 11, 12, 20, 21],
        "s_a":     [0.9, 0.5, 0.1, 0.2, 0.8],
        "s_b":     [0.1, 0.5, 0.9, 0.7, 0.3],
    })


def test_rank_within_session(cands):
    assert rank_within_session(cands, "s_a").tolist() == [1, 2, 3, 2, 1]


def test_blend_is_weighted_reciprocal_rank(cands):
    blend = blend_scores(cands, {"clicks": "s_a", "orders": "s_b"}, {"clicks": 0.1, "orders": 0.6})
    # aid 10: rank 1 on s_a, rank 3 on s_b
    assert blend.iloc[0] == pytest.approx(0.1 / 1 + 0.6 / 3)
    # aid 12: rank 3 on s_a, rank 1 on s_b
    assert blend.iloc[2] == pytest.approx(0.1 / 3 + 0.6 / 1)


def test_top_k_lists_order_and_length(cands):
    lists = top_k_lists(cands, "s_a", k=2)
    assert lists.loc[1] == [10, 11]
    assert lists.loc[2] == [21, 20]


def test_otto_recall_hand_computed():
    labels = pd.DataFrame({
        "session": [1, 2, 3],
        "click": [10, 99, None],
        "carts": [[10, 11], None, [30]],
        "orders": [[12], None, None],
    })
    lists = {
        "clicks": pd.Series({1: [10, 11], 2: [20]}),   # session 3 has no list
        "carts": pd.Series({1: [11], 2: [20], 3: [31]}),
        "orders": pd.Series({1: [12], 2: [20], 3: [30]}),
    }
    r = otto_recall(lists, labels)
    assert r["clicks"] == pytest.approx(1 / 2)        # session 1 hit, session 2 miss
    assert r["carts"] == pytest.approx(1 / 3)         # 1 hit of (2 + 1) answers
    assert r["orders"] == pytest.approx(1.0)          # 1 of 1
    assert r["score"] == pytest.approx(0.1 * 0.5 + 0.3 / 3 + 0.6 * 1.0)


def test_missing_session_counts_as_miss():
    labels = pd.DataFrame({"session": [1], "click": [5], "carts": [[5]], "orders": [[5]]})
    empty = pd.Series(dtype=object)
    r = otto_recall({"clicks": empty, "carts": empty, "orders": empty}, labels)
    assert r["clicks"] == 0 and r["carts"] == 0 and r["orders"] == 0


def test_denominator_capped_at_20():
    truth = list(range(30))
    labels = pd.DataFrame({"session": [1], "click": [0], "carts": [truth], "orders": [truth]})
    one = pd.Series({1: list(range(20))})
    r = otto_recall({"clicks": one, "carts": one, "orders": one}, labels)
    assert r["carts"] == pytest.approx(1.0)
    assert not math.isnan(r["score"])

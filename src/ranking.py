"""Turn ranker scores into top-k lists and score them with OTTO's Recall@20 (pandas).

Pure pandas/NumPy, so it runs on saved feature files without Spark.
Mirrors src/metrics.py: sessions without a list count as misses, and carts/orders
recall divides by min(number of right answers, 20).
"""
import numpy as np
import pandas as pd

TYPES = ("clicks", "carts", "orders")
WEIGHTS = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
TARGET_COLS = {"clicks": "y_click", "carts": "y_cart", "orders": "y_order"}

FEATURES = [
    "hist_rank", "cov_score", "cov_rank", "pop_rank",
    "n_clicks", "n_carts", "n_orders",
    "sess_len", "sess_items",
    "item_clicks", "item_carts", "item_orders",
    "secs_since",
    "gru_rank", "gru_sim", "sim_all",
]

# v1: the original system. v2: adds the viewed-together (click co-visitation) source.
FEATURE_SETS = {
    "v1": FEATURES,
    "v2": FEATURES + ["clk_score", "clk_rank"],
}


def rank_within_session(d: pd.DataFrame, score_col: str) -> pd.Series:
    """Rank of each candidate inside its session: 1 = highest score, ties by row order."""
    return d.groupby("session")[score_col].rank(ascending=False, method="first")


def blend_scores(d: pd.DataFrame, score_cols: dict[str, str], weights: dict[str, float]) -> pd.Series:
    """Weighted reciprocal-rank blend of several rankers (higher is better).

    This is how the original system merged the click, cart and order rankers
    into a single list.
    """
    blend = pd.Series(0.0, index=d.index)
    for t, col in score_cols.items():
        blend += weights[t] / rank_within_session(d, col)
    return blend


def top_k_lists(d: pd.DataFrame, score_col: str, k: int = 20) -> pd.Series:
    """Series indexed by session: the k highest-scoring aids, best first."""
    top = d.sort_values(["session", score_col], ascending=[True, False], kind="stable")
    top = top.groupby("session", sort=True).head(k)
    return top.groupby("session", sort=True)["aid"].agg(lambda a: [int(x) for x in a])


def _items(x) -> list[int]:
    """Label cell -> list of ints (None / NaN / missing -> empty list)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    return [int(v) for v in x]


def otto_recall(lists: dict[str, pd.Series], labels: pd.DataFrame) -> dict[str, float]:
    """OTTO Recall@20 per event type plus the weighted score.

    lists: event type -> Series(session -> list of aids). Pass the same Series
        for all three types to score a single list.
    labels: columns session, click (int or missing), carts, orders (lists or missing).
    """
    lab = labels.set_index("session")
    result = {}

    clicks = lab["click"].dropna().astype("int64")
    pred = lists["clicks"].reindex(clicks.index)
    hits = [int(c) in set(_items(p)) for c, p in zip(clicks.to_numpy(), pred.to_numpy())]
    result["clicks"] = float(np.mean(hits)) if hits else float("nan")

    for t in ("carts", "orders"):
        pred = lists[t].reindex(lab.index)
        n_hits = n_denom = 0
        for truth, p in zip(lab[t].to_numpy(), pred.to_numpy()):
            truth = set(_items(truth))
            if not truth:
                continue
            n_hits += len(truth & set(_items(p)))
            n_denom += min(len(truth), 20)
        result[t] = n_hits / n_denom if n_denom else float("nan")

    result["score"] = sum(WEIGHTS[t] * result[t] for t in TYPES)
    return result

"""src/serve.py end to end, on tiny models and tables built in a temporary folder."""
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import torch

from src.model import SessionGRU
from src.ranking import TYPES
from src.serve import FEATURES, GRU_FILE, RANKER_FILES, Recommender, covisit_candidates

N_ITEMS = 50
POPULAR = np.arange(10, 30, dtype=np.int32)


def csr(partners: dict[int, list[tuple[int, int]]]) -> dict[str, np.ndarray]:
    """{item: [(partner, count), ...]} -> CSR arrays like scripts.export_serving writes."""
    offsets = np.zeros(N_ITEMS + 1, dtype=np.int64)
    flat_p, flat_c = [], []
    for a in range(N_ITEMS):
        for p, c in partners.get(a, []):
            flat_p.append(p)
            flat_c.append(c)
        offsets[a + 1] = len(flat_p)
    return {"offsets": offsets, "partners": np.array(flat_p, dtype=np.int32),
            "counts": np.array(flat_c, dtype=np.int32)}


@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    root = tmp_path_factory.mktemp("serving")
    artifacts, models = root / "artifacts", root / "models"
    artifacts.mkdir()
    models.mkdir()

    np.savez(artifacts / "covisit_buys.npz", **csr({1: [(2, 5), (3, 1)], 2: [(1, 2)]}))
    np.savez(artifacts / "covisit_clicks.npz", **csr({1: [(4, 7)]}))
    np.save(artifacts / "item_pop.npy", np.arange(N_ITEMS * 3, dtype=np.int32).reshape(N_ITEMS, 3))
    np.save(artifacts / "popular.npy", POPULAR)

    torch.manual_seed(0)
    torch.save(SessionGRU(N_ITEMS, dim=8).state_dict(), models / GRU_FILE)

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, len(FEATURES))), columns=FEATURES)
    for t, name in RANKER_FILES.items():
        booster = lgb.train({"objective": "regression", "num_leaves": 4, "verbose": -1},
                            lgb.Dataset(X, rng.normal(size=200)), num_boost_round=5)
        booster.save_model(str(models / name))

    return Recommender(artifacts, models, gru_k=10)


EVENTS = [
    {"aid": 2, "ts": 100, "type": 0},
    {"aid": 1, "ts": 200, "type": 0},
    {"aid": 1, "ts": 250, "type": 1},
]


def test_three_lists_of_k_unique_items(rec):
    lists = rec.recommend(EVENTS, k=5)
    assert set(lists) == set(TYPES)
    for items in lists.values():
        assert len(items) == 5
        assert len(set(items)) == 5
        assert all(isinstance(a, int) and 0 <= a < N_ITEMS for a in items)


def test_empty_session_gets_popular_items(rec):
    lists = rec.recommend([], k=3)
    assert lists == {t: [10, 11, 12] for t in TYPES}


def test_features_for_history_and_covisitation(rec):
    f = rec.features(EVENTS).set_index("aid")
    assert set(FEATURES) <= set(f.columns)
    assert f.index.is_unique

    # history: most recent first; secs_since from the last event of the session
    assert f.loc[1, "hist_rank"] == 1 and f.loc[2, "hist_rank"] == 2
    assert f.loc[1, "secs_since"] == 0 and f.loc[2, "secs_since"] == 150
    assert (f.loc[1, "n_clicks"], f.loc[1, "n_carts"], f.loc[1, "n_orders"]) == (1, 1, 0)

    # bought-together partners of items 1 and 2, summed over the history
    assert f.loc[2, "cov_score"] == 5 and f.loc[1, "cov_score"] == 2 and f.loc[3, "cov_score"] == 1
    # viewed-together partner of item 1
    assert f.loc[4, "clk_score"] == 7 and f.loc[4, "clk_rank"] == 1

    # every popular item is a candidate; session-level features are filled everywhere
    assert set(POPULAR) <= set(f.index)
    assert (f["sess_len"] == 3).all() and (f["sess_items"] == 2).all()
    # item activity comes from the lookup table
    assert f.loc[4, "item_clicks"] == 12 and f.loc[4, "item_orders"] == 14


def test_covisit_ties_go_to_the_lower_item_id():
    table = tuple(csr({0: [(9, 3), (5, 3)], 1: [(7, 1)]}).values())
    df = covisit_candidates(table, np.array([0, 1]), "s", "r")
    assert df["aid"].tolist() == [5, 9, 7]
    assert df["r"].tolist() == [1, 2, 3]


def test_no_partners_gives_an_empty_table():
    table = tuple(csr({}).values())
    assert covisit_candidates(table, np.array([3]), "s", "r").empty

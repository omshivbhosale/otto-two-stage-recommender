"""Two-stage recommender for a single session: candidates -> features -> rankers -> top-k lists.

Mirrors the v2 system (src/features.py, Spark) with NumPy/pandas so it can answer in
milliseconds: five candidate sources, 18 features, and one LightGBM ranker per event
type, each producing its own list. scripts.check_serving compares it with the pipeline.
"""
import os
from pathlib import Path

# torch and lightgbm each bundle their own OpenMP threading library. On macOS,
# running both multi-threaded in one process can freeze or crash. Serving handles
# one session at a time, so a single thread is fast enough and avoids the clash.
# These must be set BEFORE torch and lightgbm are imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: E402,I001  (must come before lightgbm)

torch.set_num_threads(1)

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.model import SessionGRU  # noqa: E402
from src.ranking import FEATURE_SETS, TARGET_COLS, TYPES  # noqa: E402
from src.retrieval import build_sequences  # noqa: E402

FEATURES = FEATURE_SETS["v2"]
N_AIDS = 1_855_603
GRU_FILE = "gru_test.pt"
RANKER_FILES = {t: f"ranker_v2_{col}.txt" for t, col in TARGET_COLS.items()}


def load_covisit(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR co-visitation table written by scripts.export_serving."""
    cv = np.load(path)
    return cv["offsets"], cv["partners"], cv["counts"]


def covisit_candidates(
    table: tuple[np.ndarray, np.ndarray, np.ndarray], hist_aids: np.ndarray, score: str, rank: str, k: int = 20
) -> pd.DataFrame:
    """Partners of the history items, summed over the history, top k by score (ties: lower aid first)."""
    offsets, partners, counts = table
    spans = [np.arange(offsets[a], offsets[a + 1]) for a in hist_aids]
    idx = np.concatenate(spans) if spans else np.array([], dtype=np.int64)
    if len(idx) == 0:
        return pd.DataFrame({"aid": pd.Series(dtype="int64"), score: pd.Series(dtype="int64"),
                             rank: pd.Series(dtype="int64")})
    df = pd.DataFrame({"aid": partners[idx].astype(np.int64), score: counts[idx].astype(np.int64)})
    df = df.groupby("aid", as_index=False)[score].sum()
    df = df.sort_values([score, "aid"], ascending=[False, True], kind="stable").head(k)
    df[rank] = np.arange(1, len(df) + 1)
    return df


class Recommender:
    def __init__(self, artifacts: Path, models: Path, gru_k: int = 40):
        self.buys = load_covisit(artifacts / "covisit_buys.npz")
        self.clicks = load_covisit(artifacts / "covisit_clicks.npz")
        self.item_pop = np.load(artifacts / "item_pop.npy")
        self.popular = np.load(artifacts / "popular.npy")

        state = torch.load(models / GRU_FILE, map_location="cpu")
        n_items = state["emb.weight"].shape[0] - 1
        self.model = SessionGRU(n_items, dim=state["emb.weight"].shape[1])
        self.model.load_state_dict(state)
        self.model.eval()
        with torch.no_grad():
            self.item_vecs = self.model.item_vectors()  # (n_items + 1, dim), row 0 = padding

        self.boosters = {t: lgb.Booster(model_file=str(models / f)) for t, f in RANKER_FILES.items()}
        self.gru_k = gru_k

    # ---------- stage 1: candidates ----------

    @staticmethod
    def _history(ev: pd.DataFrame) -> pd.DataFrame:
        last = ev.groupby("aid")["ts"].max().reset_index(name="last_ts")
        last = last.sort_values(["last_ts", "aid"], ascending=[False, True], kind="stable").head(20)
        last["hist_rank"] = np.arange(1, len(last) + 1)
        return last[["aid", "hist_rank"]]

    @torch.no_grad()
    def _gru(self, ev: pd.DataFrame) -> tuple[pd.DataFrame, torch.Tensor]:
        _, seqs = build_sequences(ev.assign(session=0))
        sv = self.model.encode(torch.from_numpy(seqs))[0]
        scores = self.item_vecs @ sv
        scores[0] = -float("inf")  # never recommend padding
        k = min(self.gru_k, len(scores) - 1)
        top = scores.topk(k)
        df = pd.DataFrame({
            "aid": top.indices.numpy().astype(np.int64) - 1,
            "gru_rank": np.arange(1, k + 1),
            "gru_sim": top.values.numpy(),
        })
        return df, sv

    # ---------- stage 2: features + ranking ----------

    def features(self, events: list[dict]) -> pd.DataFrame:
        """All candidates for one session with ranker features (one row per item)."""
        ev = pd.DataFrame(events, columns=["aid", "ts", "type"]).astype("int64")

        hist = self._history(ev)
        hist_aids = hist["aid"].to_numpy()
        cov = covisit_candidates(self.buys, hist_aids, "cov_score", "cov_rank")
        clk = covisit_candidates(self.clicks, hist_aids, "clk_score", "clk_rank")
        pop = pd.DataFrame({"aid": self.popular.astype(np.int64), "pop_rank": np.arange(1, len(self.popular) + 1)})
        gru, sv = self._gru(ev)

        cands = hist
        for part in (cov, pop, clk, gru):
            cands = cands.merge(part, on="aid", how="outer")

        inter = ev.groupby("aid").agg(
            n_clicks=("type", lambda t: int((t == 0).sum())),
            n_carts=("type", lambda t: int((t == 1).sum())),
            n_orders=("type", lambda t: int((t == 2).sum())),
            item_last_ts=("ts", "max"),
        ).reset_index()
        cands = cands.merge(inter, on="aid", how="left")

        cands["sess_len"] = len(ev)
        cands["sess_items"] = ev["aid"].nunique()
        cands["secs_since"] = ev["ts"].max() - cands["item_last_ts"]

        aids = cands["aid"].to_numpy().astype(np.int64)
        cands["item_clicks"], cands["item_carts"], cands["item_orders"] = self.item_pop[aids].T

        cands[["n_clicks", "n_carts", "n_orders"]] = cands[["n_clicks", "n_carts", "n_orders"]].fillna(0)
        cands["sim_all"] = (self.item_vecs[aids + 1] @ sv).numpy()
        return cands.drop(columns=["item_last_ts"])

    def score(self, cands: pd.DataFrame) -> pd.DataFrame:
        """Add one score column per event type."""
        for t in TYPES:
            cands[f"score_{t}"] = self.boosters[t].predict(cands[FEATURES])
        return cands

    def recommend(self, events: list[dict], k: int = 20) -> dict[str, list[int]]:
        """Up to k item IDs per event type. Events: [{"aid", "ts", "type"}, ...]."""
        if not events:
            fallback = [int(a) for a in self.popular[:k]]
            return {t: list(fallback) for t in TYPES}
        cands = self.score(self.features(events))
        aids = cands["aid"].to_numpy()
        return {
            t: [int(a) for a in aids[np.argsort(-cands[f"score_{t}"].to_numpy(), kind="stable")[:k]]]
            for t in TYPES
        }

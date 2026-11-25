"""Two-stage recommender for a single session: candidates -> features -> rankers -> top 20.

Mirrors src/features.py (Spark) with NumPy/pandas so it can answer in milliseconds.
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

from src.model import SessionGRU
from src.retrieval import build_sequences

FEATURES = [
    "hist_rank", "cov_score", "cov_rank", "pop_rank",
    "n_clicks", "n_carts", "n_orders",
    "sess_len", "sess_items",
    "item_clicks", "item_carts", "item_orders",
    "secs_since",
    "gru_rank", "gru_sim", "sim_all",
]
TARGETS = ["y_click", "y_cart", "y_order"]
WEIGHTS = {"y_click": 0.1, "y_cart": 0.3, "y_order": 0.6}
N_AIDS = 1_855_603


class Recommender:
    def __init__(self, artifacts: Path, models: Path, gru_k: int = 40):
        cv = np.load(artifacts / "covisit.npz")
        self.cv_offsets, self.cv_partners, self.cv_counts = cv["offsets"], cv["partners"], cv["counts"]
        self.item_pop = np.load(artifacts / "item_pop.npy")
        self.popular = np.load(artifacts / "popular.npy")

        self.model = SessionGRU(N_AIDS, dim=64)
        self.model.load_state_dict(torch.load(models / "session_gru_epoch1.pt", map_location="cpu"))
        self.model.eval()
        with torch.no_grad():
            self.item_vecs = self.model.item_vectors()  # (N_AIDS + 1, 64), row 0 = padding

        self.boosters = {y: lgb.Booster(model_file=str(models / f"ranker_{y}.txt")) for y in TARGETS}
        self.gru_k = gru_k

    # ---------- stage 1: candidates ----------

    def _history(self, ev: pd.DataFrame) -> pd.DataFrame:
        last = ev.groupby("aid")["ts"].max().reset_index(name="last_ts")
        last = last.sort_values(["last_ts", "aid"], ascending=[False, True]).head(20)
        last["hist_rank"] = np.arange(1, len(last) + 1)
        return last[["aid", "hist_rank"]]

    def _covisit(self, hist_aids: np.ndarray) -> pd.DataFrame:
        parts, counts = [], []
        for a in hist_aids:
            s, e = self.cv_offsets[a], self.cv_offsets[a + 1]
            parts.append(self.cv_partners[s:e])
            counts.append(self.cv_counts[s:e])
        if not parts or sum(len(p) for p in parts) == 0:
            return pd.DataFrame({"aid": [], "cov_score": [], "cov_rank": []})
        df = pd.DataFrame({"aid": np.concatenate(parts), "count": np.concatenate(counts)})
        df = df.groupby("aid")["count"].sum().reset_index(name="cov_score")
        df = df.sort_values(["cov_score", "aid"], ascending=[False, True]).head(20)
        df["cov_rank"] = np.arange(1, len(df) + 1)
        return df

    @torch.no_grad()
    def _gru(self, ev: pd.DataFrame) -> tuple[pd.DataFrame, torch.Tensor]:
        _, seqs = build_sequences(ev.assign(session=0))
        sv = self.model.encode(torch.from_numpy(seqs))[0]
        scores = self.item_vecs @ sv
        scores[0] = -float("inf")
        top = scores.topk(self.gru_k)
        df = pd.DataFrame({
            "aid": top.indices.numpy() - 1,
            "gru_rank": np.arange(1, self.gru_k + 1),
            "gru_sim": top.values.numpy(),
        })
        return df, sv

    # ---------- stage 2: features + ranking ----------

    def features(self, events: list[dict]) -> pd.DataFrame:
        """All candidates for one session with ranker features (one row per item)."""
        ev = pd.DataFrame(events, columns=["aid", "ts", "type"]).astype("int64")

        hist = self._history(ev)
        cov = self._covisit(hist["aid"].to_numpy())
        pop = pd.DataFrame({"aid": self.popular, "pop_rank": np.arange(1, len(self.popular) + 1)})
        gru, sv = self._gru(ev)

        cands = hist
        for part in (cov, pop, gru):
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
        pop_counts = self.item_pop[aids]
        cands["item_clicks"], cands["item_carts"], cands["item_orders"] = pop_counts.T

        cands[["n_clicks", "n_carts", "n_orders"]] = cands[["n_clicks", "n_carts", "n_orders"]].fillna(0)
        cands["sim_all"] = (self.item_vecs[aids + 1] @ sv).numpy()
        return cands.drop(columns=["item_last_ts"])

    def recommend(self, events: list[dict], k: int = 20) -> list[int]:
        """Top-k item IDs for one session. Events: [{"aid", "ts", "type"}, ...]."""
        if not events:
            return [int(a) for a in self.popular[:k]]
        cands = self.features(events)
        blend = np.zeros(len(cands))
        for y in TARGETS:
            score = self.boosters[y].predict(cands[FEATURES])
            rank = pd.Series(score).rank(ascending=False, method="first").to_numpy()
            blend += WEIGHTS[y] / rank
        order = np.argsort(-blend, kind="stable")[:k]
        return [int(a) for a in cands["aid"].to_numpy()[order]]
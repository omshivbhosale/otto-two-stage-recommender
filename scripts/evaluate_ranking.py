"""Compare one blended list against separate click/cart/order lists (OTTO Recall@20).

Uses the saved LightGBM rankers and saved feature files; no Spark, no retraining.

  python -m scripts.evaluate_ranking              # ranker week, held-out sessions (use this to decide)
  python -m scripts.evaluate_ranking --final      # test week: run once, at the very end

Ranker week: sessions the rankers never trained on (everything outside the
300K-session training sample), so the comparison is fair.
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from src.ranking import FEATURES, TARGET_COLS, TYPES, WEIGHTS, blend_scores, otto_recall, top_k_lists

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

RAW_FEATURES = [f for f in FEATURES if f != "sim_all"]  # sim_all is computed here from saved vectors


def load_boosters() -> dict[str, lgb.Booster]:
    return {t: lgb.Booster(model_file=str(MODELS / f"ranker_{col}.txt")) for t, col in TARGET_COLS.items()}


def add_sim_all(d: pd.DataFrame, prefix: str) -> None:
    """GRU similarity between each session and each candidate (saved vectors, IDs shifted +1)."""
    sess_ids = np.load(DATA / f"{prefix}_session_ids.npy")
    sess_vecs = np.load(DATA / f"{prefix}_session_vecs.npy")
    item_vecs = np.load(DATA / f"{prefix}_item_vecs.npy")
    s_idx = np.searchsorted(sess_ids, d["session"].to_numpy())
    assert (sess_ids[s_idx] == d["session"].to_numpy()).all(), "session vector missing"
    i_idx = d["aid"].to_numpy() + 1
    sim = np.empty(len(d), dtype=np.float32)
    step = 2_000_000
    for i in range(0, len(d), step):
        sim[i:i + step] = (sess_vecs[s_idx[i:i + step]] * item_vecs[i_idx[i:i + step]]).sum(axis=1)
    d["sim_all"] = sim


def strategy_lists(d: pd.DataFrame, boosters: dict[str, lgb.Booster]) -> dict[str, dict[str, pd.Series]]:
    """Both strategies for one chunk of candidates: one blended list vs one list per type."""
    for t, booster in boosters.items():
        d[f"score_{t}"] = booster.predict(d[FEATURES])
    d["blend"] = blend_scores(d, {t: f"score_{t}" for t in TYPES}, WEIGHTS)
    one = top_k_lists(d, "blend")
    return {
        "one blended list (current)": {t: one for t in TYPES},
        "separate list per type": {t: top_k_lists(d, f"score_{t}") for t in TYPES},
    }


def merge(acc: dict, new: dict) -> None:
    for name, by_type in new.items():
        for t, s in by_type.items():
            acc.setdefault(name, {}).setdefault(t, []).append(s)


def ranker_week(n_sessions: int, boosters, seed: int) -> tuple[dict, pd.DataFrame]:
    labels = pd.read_parquet(DATA / "rank_labels")
    trained_on = pd.read_parquet(DATA / "rank_sample_gru", columns=["session"])["session"].unique()
    pool = np.setdiff1d(labels["session"].to_numpy(), trained_on)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(pool, size=min(n_sessions, len(pool)), replace=False))
    print(f"held-out sessions: {len(pool):,} | evaluating {len(chosen):,} (seed {seed})")

    dataset = ds.dataset(DATA / "rank_features_gru", format="parquet")
    session_type = dataset.schema.field("session").type
    table = dataset.to_table(
        columns=["session", "aid", *RAW_FEATURES],
        filter=ds.field("session").isin(pa.array(chosen).cast(session_type)),
    )
    d = table.to_pandas()
    print(f"candidate rows: {len(d):,}")
    add_sim_all(d, "rank")

    acc: dict = {}
    merge(acc, strategy_lists(d, boosters))
    return acc, labels[labels["session"].isin(chosen)]


def test_week(boosters) -> tuple[dict, pd.DataFrame]:
    acc: dict = {}
    start = time.time()
    for b in range(20):
        d = pd.read_parquet(DATA / "test_features" / f"bucket={b}", columns=["session", "aid", *RAW_FEATURES])
        add_sim_all(d, "test")
        merge(acc, strategy_lists(d, boosters))
        print(f"bucket {b + 1}/20 | {len(d):,} rows | {(time.time() - start) / 60:.1f} min", flush=True)
    return acc, pd.read_parquet(DATA / "test_labels")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final", action="store_true", help="score the test week (run once, at the end)")
    parser.add_argument("--sessions", type=int, default=200_000, help="ranker-week sessions to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start = time.time()
    boosters = load_boosters()
    week = "test" if args.final else "ranker"
    acc, labels = test_week(boosters) if args.final else ranker_week(args.sessions, boosters, args.seed)

    rows = {}
    for name, by_type in acc.items():
        lists = {t: pd.concat(parts) for t, parts in by_type.items()}
        rows[name] = otto_recall(lists, labels)
    table = pd.DataFrame(rows).T[["clicks", "carts", "orders", "score"]].round(4)
    print(f"\nOTTO Recall@20, {week} week, {len(labels):,} sessions\n")
    print(table.to_string())

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"ranking_strategies_{week}_week.json"
    out.write_text(json.dumps({"week": week, "sessions": int(len(labels)), "results": rows}, indent=2))
    print(f"\nsaved {out.relative_to(ROOT)} | {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()

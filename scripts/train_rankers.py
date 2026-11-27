"""Train the click, cart and order rankers on the ranker week (LightGBM lambdarank).

  python -m scripts.train_rankers --version v1
  python -m scripts.train_rankers --version v2

All versions train on the same 300K ranker-week sessions (80% to train, 20% to
stop early), chosen once at random among sessions with at least one right answer
in the v1 candidate pool. Every other ranker-week session is held out, so
scripts.evaluate_ranking can compare versions on sessions no ranker has seen.
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from scripts._common import (
    DATA, FEATURE_DIRS, REPORTS, TRAINING_SESSIONS, add_sim_all, ranker_path, read_sessions, training_sessions,
)
from src.ranking import FEATURE_SETS, TARGET_COLS

PARAMS = dict(
    objective="lambdarank", n_estimators=300, learning_rate=0.05,
    num_leaves=63, min_child_samples=50, verbose=-1,
)


def with_positives(d: pd.DataFrame, y: str) -> pd.DataFrame:
    """Sessions with at least one right answer among the candidates (the only ones a ranker can learn from)."""
    return d[d.groupby("session")[y].transform("max") > 0]


def pool_recall(d: pd.DataFrame, score_col: str, y: str) -> float:
    """Share of the right answers in the candidate pool that land in the top 20."""
    rk = d.groupby("session")[score_col].rank(ascending=False, method="first")
    hits = d[rk <= 20].groupby("session")[y].sum()
    pos = d.groupby("session")[y].sum()
    pos = pos[pos > 0]
    return float(hits.reindex(pos.index, fill_value=0).sum() / np.minimum(pos, 20).sum())


def choose_training_sessions(n: int = 300_000, seed: int = 1) -> None:
    """Pick the training sessions once and save them; later runs reuse the same ones."""
    if TRAINING_SESSIONS.exists():
        return
    cols = ["session", *TARGET_COLS.values()]
    d = ds.dataset(DATA / FEATURE_DIRS[("v1", "rank")], format="parquet").to_table(columns=cols).to_pandas()
    has_answer = d.groupby("session")[list(TARGET_COLS.values())].max().sum(axis=1) > 0
    candidates = np.sort(has_answer[has_answer].index.to_numpy())
    rng = np.random.default_rng(seed)
    np.save(TRAINING_SESSIONS, np.sort(rng.choice(candidates, size=n, replace=False)))
    print(f"chose {n:,} training sessions out of {len(candidates):,} with a right answer in the pool")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", choices=FEATURE_SETS, default="v2")
    args = parser.parse_args()
    start = time.time()

    choose_training_sessions()
    features = FEATURE_SETS[args.version]
    raw = [f for f in features if f != "sim_all"]
    d = read_sessions(
        DATA / FEATURE_DIRS[(args.version, "rank")],
        training_sessions(),
        ["session", "aid", *raw, *TARGET_COLS.values()],
    )
    add_sim_all(d, "rank")
    print(f"rows: {len(d):,} | sessions: {d['session'].nunique():,} | "
          f"memory: {d.memory_usage(deep=True).sum() / 1e9:.2f} GB", flush=True)

    # same 80/20 split of sessions as the original rankers
    rng = np.random.default_rng(0)
    all_sessions = d["session"].unique()
    valid_sessions = rng.choice(all_sessions, size=len(all_sessions) // 5, replace=False)
    is_valid = d["session"].isin(valid_sessions)
    train_df, valid_df = d[~is_valid], d[is_valid].copy()

    report = {"version": args.version, "features": features, "targets": {}}
    for t, y in TARGET_COLS.items():
        tr, va = with_positives(train_df, y), with_positives(valid_df, y)
        model = lgb.LGBMRanker(**PARAMS)
        model.fit(
            tr[features], tr[y], group=tr.groupby("session", sort=False).size().to_numpy(),
            eval_set=[(va[features], va[y])],
            eval_group=[va.groupby("session", sort=False).size().to_numpy()],
            eval_at=[20],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        model.booster_.save_model(str(ranker_path(args.version, y)))
        valid_df[f"score_{t}"] = model.predict(valid_df[features])
        gain = pd.Series(model.booster_.feature_importance("gain"), index=features)
        report["targets"][t] = {
            "trees": int(model.best_iteration_ or PARAMS["n_estimators"]),
            "pool_recall_at_20": pool_recall(valid_df, f"score_{t}", y),
            "gain_share": (gain / gain.sum()).round(4).sort_values(ascending=False).to_dict(),
        }
        print(f"{t}: trees {report['targets'][t]['trees']} | "
              f"pool recall@20 {report['targets'][t]['pool_recall_at_20']:.4f} | "
              f"{(time.time() - start) / 60:.1f} min", flush=True)

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"rankers_{args.version}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"saved {out.name} and models/{ranker_path(args.version, 'y_*').name}")


if __name__ == "__main__":
    main()

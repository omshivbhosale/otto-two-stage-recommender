"""Score systems with OTTO Recall@20: one blended list vs separate click/cart/order lists.

Uses saved rankers and saved feature files; no Spark, no retraining.

  python -m scripts.evaluate_ranking                          # v1, ranker week (held-out sessions)
  python -m scripts.evaluate_ranking --versions v1 v2         # compare systems on the same sessions
  python -m scripts.evaluate_ranking --versions v2 --final    # test week: run once, at the very end

Ranker week: sessions outside the 300K the rankers trained on, so the comparison is fair.
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts._common import (
    DATA, FEATURE_DIRS, REPORTS, ROOT, add_sim_all, ranker_path, read_sessions, training_sessions,
)
from src.ranking import FEATURE_SETS, TARGET_COLS, TYPES, WEIGHTS, blend_scores, otto_recall, top_k_lists


def load_boosters(version: str) -> dict[str, lgb.Booster]:
    return {t: lgb.Booster(model_file=str(ranker_path(version, col))) for t, col in TARGET_COLS.items()}


def strategy_lists(d: pd.DataFrame, boosters: dict, features: list[str]) -> dict[str, dict[str, pd.Series]]:
    """Both strategies for one chunk of candidates: one blended list vs one list per type."""
    for t, booster in boosters.items():
        d[f"score_{t}"] = booster.predict(d[features])
    d["blend"] = blend_scores(d, {t: f"score_{t}" for t in TYPES}, WEIGHTS)
    one = top_k_lists(d, "blend")
    return {
        "one blended list": {t: one for t in TYPES},
        "separate list per type": {t: top_k_lists(d, f"score_{t}") for t in TYPES},
    }


def merge(acc: dict, new: dict) -> None:
    for name, by_type in new.items():
        for t, s in by_type.items():
            acc.setdefault(name, {}).setdefault(t, []).append(s)


def score_ranker_week(version: str, chosen: np.ndarray) -> dict:
    features = FEATURE_SETS[version]
    raw = [f for f in features if f != "sim_all"]
    d = read_sessions(DATA / FEATURE_DIRS[(version, "rank")], chosen, ["session", "aid", *raw])
    print(f"{version}: {len(d):,} candidate rows ({len(d) / len(chosen):.1f} per session)", flush=True)
    add_sim_all(d, "rank")
    acc: dict = {}
    merge(acc, strategy_lists(d, load_boosters(version), features))
    return acc


def score_test_week(version: str) -> dict:
    features = FEATURE_SETS[version]
    raw = [f for f in features if f != "sim_all"]
    boosters, acc, start = load_boosters(version), {}, time.time()
    for b in range(20):
        d = pd.read_parquet(DATA / FEATURE_DIRS[(version, "test")] / f"bucket={b}", columns=["session", "aid", *raw])
        add_sim_all(d, "test")
        merge(acc, strategy_lists(d, boosters, features))
        print(f"{version} bucket {b + 1}/20 | {len(d):,} rows | {(time.time() - start) / 60:.1f} min", flush=True)
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--versions", nargs="+", choices=FEATURE_SETS, default=["v1"])
    parser.add_argument("--final", action="store_true", help="score the test week (run once, at the end)")
    parser.add_argument("--sessions", type=int, default=200_000, help="ranker-week sessions to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    start = time.time()

    if args.final:
        week = "test"
        labels = pd.read_parquet(DATA / "test_labels")
    else:
        week = "ranker"
        all_labels = pd.read_parquet(DATA / "rank_labels")
        pool = np.setdiff1d(all_labels["session"].to_numpy(), training_sessions())
        rng = np.random.default_rng(args.seed)
        chosen = np.sort(rng.choice(pool, size=min(args.sessions, len(pool)), replace=False))
        labels = all_labels[all_labels["session"].isin(chosen)]
        print(f"held-out sessions: {len(pool):,} | evaluating {len(chosen):,} (seed {args.seed})")

    rows = {}
    for version in args.versions:
        acc = score_test_week(version) if args.final else score_ranker_week(version, chosen)
        for name, by_type in acc.items():
            lists = {t: pd.concat(parts) for t, parts in by_type.items()}
            rows[f"{version}, {name}"] = otto_recall(lists, labels)

    table = pd.DataFrame(rows).T[["clicks", "carts", "orders", "score"]].round(4)
    print(f"\nOTTO Recall@20, {week} week, {len(labels):,} sessions\n")
    print(table.to_string())

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"evaluation_{week}_week_{'_'.join(args.versions)}.json"
    out.write_text(json.dumps(
        {"week": week, "sessions": int(len(labels)), "seed": None if args.final else args.seed, "results": rows},
        indent=2,
    ))
    print(f"\nsaved {out.relative_to(ROOT)} | {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()

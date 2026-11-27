"""Check that the API code (src/serve.py) reproduces the pipeline on real test-week sessions.

  python -m scripts.check_serving               # 200 random test-week sessions
  python -m scripts.check_serving --sessions 1000

For each session, the events before the cut go through src/serve.py one at a time,
exactly as the API receives them. The result is compared with the saved test-week
features (Spark) scored by the same rankers:

  - candidates: the same set of items
  - features: integer features equal, GRU similarities within a small tolerance
    (the pipeline computed them on the Mac GPU, serving on the CPU)
  - lists: the same top 20 per event type

Also measures latency. Writes reports/serving_check.json.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from scripts._common import ARTIFACTS, DATA, FEATURE_DIRS, MODELS, REPORTS, add_sim_all, read_sessions
from src.ranking import TYPES
from src.serve import FEATURES, Recommender

FLOAT_FEATURES = ["gru_sim", "sim_all"]
TOLERANCE = 1e-3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    start = time.time()

    inp = pd.read_parquet(DATA / "test_input", columns=["session", "aid", "ts", "type"])
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(inp["session"].unique(), size=args.sessions, replace=False))
    inp = inp[inp["session"].isin(chosen)].sort_values(["session", "ts"], kind="stable")

    pipe = read_sessions(DATA / FEATURE_DIRS[("v2", "test")], chosen,
                         ["session", "aid", *[f for f in FEATURES if f != "sim_all"]])
    add_sim_all(pipe, "test")
    rec = Recommender(ARTIFACTS, MODELS)
    pipe = rec.score(pipe)
    print(f"loaded {len(chosen)} sessions | {(time.time() - start) / 60:.1f} min", flush=True)

    same_cands = 0
    same_lists = {t: 0 for t in TYPES}
    int_mismatch = 0
    max_float_diff = 0.0
    latencies = []
    for session, ev in inp.groupby("session", sort=True):
        events = ev[["aid", "ts", "type"]].astype(int).to_dict("records")

        t0 = time.perf_counter()
        lists = rec.recommend(events, k=20)
        latencies.append((time.perf_counter() - t0) * 1000)

        served = rec.features(events).set_index("aid").sort_index()
        saved = pipe[pipe["session"] == session].set_index("aid").sort_index()
        if set(served.index) == set(saved.index):
            same_cands += 1
            for f in FEATURES:
                a, b = served[f].to_numpy(dtype=float), saved.loc[served.index, f].to_numpy(dtype=float)
                both = ~np.isnan(a) & ~np.isnan(b)
                if (np.isnan(a) != np.isnan(b)).any():
                    int_mismatch += 1
                elif f in FLOAT_FEATURES:
                    if both.any():
                        max_float_diff = max(max_float_diff, float(np.abs(a[both] - b[both]).max()))
                elif (a[both] != b[both]).any():
                    int_mismatch += 1

        for t in TYPES:
            order = np.argsort(-saved[f"score_{t}"].to_numpy(), kind="stable")[:20]
            if lists[t] == [int(x) for x in saved.index.to_numpy()[order]]:
                same_lists[t] += 1

    n = len(chosen)
    report = {
        "sessions": n,
        "seed": args.seed,
        "same_candidates": same_cands,
        "feature_mismatches": int_mismatch,
        "max_gru_similarity_diff": round(max_float_diff, 6),
        "same_top20": same_lists,
        "latency_ms": {
            "median": round(float(np.median(latencies)), 1),
            "p95": round(float(np.percentile(latencies, 95)), 1),
        },
    }
    print(json.dumps(report, indent=2))
    ok = same_cands == n and int_mismatch == 0 and max_float_diff <= TOLERANCE
    print("serving matches the pipeline" if ok else "MISMATCH: serving differs from the pipeline, see above")
    (REPORTS / "serving_check.json").write_text(json.dumps(report, indent=2))
    print(f"saved reports/serving_check.json | {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()

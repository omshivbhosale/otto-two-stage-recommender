"""GRU candidates and vectors for one week (PyTorch).

  python -m scripts.gru_candidates --week rank
  python -m scripts.gru_candidates --week test

For every session in the week's input: the 40 closest items to the session vector
(candidates for the ranker), plus the session and item vectors used for the
`sim_all` feature. Writes {week}_gru_candidates/, {week}_session_ids.npy,
{week}_session_vecs.npy and {week}_item_vecs.npy to data/processed/.
"""
import argparse
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from scripts._common import DATA, MODELS, N_ITEMS
from src.model import SessionGRU
from src.retrieval import build_sequences, encode_sessions, search

K = 40


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--week", choices=["rank", "test"], required=True)
    args = parser.parse_args()
    week, start = args.week, time.time()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SessionGRU(N_ITEMS, dim=64).to(device)
    model.load_state_dict(torch.load(MODELS / f"gru_{week}.pt", map_location=device))
    model.eval()

    inp = pd.read_parquet(DATA / f"{week}_input", columns=["session", "aid", "ts"])
    sessions, seqs = build_sequences(inp)
    del inp
    print(f"{week} sessions: {len(sessions):,}", flush=True)

    vecs = encode_sessions(model, seqs, device=device)
    with torch.no_grad():
        item_vecs = model.item_vectors().cpu().numpy()
    np.save(DATA / f"{week}_session_ids.npy", sessions)
    np.save(DATA / f"{week}_session_vecs.npy", vecs)
    np.save(DATA / f"{week}_item_vecs.npy", item_vecs)

    ids, sims = search(model, seqs, k=K, device=device, return_scores=True)

    out_dir = DATA / f"{week}_gru_candidates"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("part-*.parquet"):
        old.unlink()
    chunk = 200_000
    for part, i in enumerate(range(0, len(sessions), chunk)):
        n = len(sessions[i:i + chunk])
        pq.write_table(pa.table({
            "session": np.repeat(sessions[i:i + chunk], K).astype(np.int32),
            "aid": ids[i:i + chunk].ravel(),
            "gru_rank": np.tile(np.arange(1, K + 1, dtype=np.int32), n),
            "gru_sim": sims[i:i + chunk].ravel(),
        }), out_dir / f"part-{part:03d}.parquet")
    print(f"saved {out_dir.name}: {len(sessions) * K:,} rows | {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()

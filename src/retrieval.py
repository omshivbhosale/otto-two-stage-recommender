"""Turn sessions into GRU inputs and search for the closest items."""
import time

import numpy as np
import pandas as pd
import torch

from src.model import SessionGRU

SEQ_LEN = 20


def build_sequences(events: pd.DataFrame, seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Events (session, aid, ts) -> (session ids, padded sequences).

    Same format as training: time order, back-to-back repeats removed,
    last `seq_len` items, IDs shifted by +1, left padding with 0.
    """
    ev = events.sort_values(["session", "ts"], kind="stable")
    s, a = ev["session"].to_numpy(), ev["aid"].to_numpy()
    repeat = np.concatenate([[False], (s[1:] == s[:-1]) & (a[1:] == a[:-1])])
    ev = ev[~repeat]
    ev = ev[ev.groupby("session").cumcount(ascending=False) < seq_len]

    sessions, first_idx, lengths = np.unique(
        ev["session"].to_numpy(), return_index=True, return_counts=True
    )
    values = ev["aid"].to_numpy().astype(np.int32) + 1
    seqs = np.zeros((len(sessions), seq_len), dtype=np.int32)
    rows = np.repeat(np.arange(len(sessions)), lengths)
    pos = np.arange(len(values)) - np.repeat(first_idx, lengths)
    seqs[rows, pos + (seq_len - np.repeat(lengths, lengths))] = values
    return sessions, seqs


@torch.no_grad()
def search(
    model: SessionGRU,
    seqs: np.ndarray,
    k: int = 40,
    batch: int = 256,
    device: torch.device | str = "cpu",
    log_every: int = 500,
) -> np.ndarray:
    """Top-k closest items per sequence, as original item IDs.

    Scores on `device`, picks the top k on the CPU (much faster than
    top-k on the Mac GPU for 1.8M items).
    """
    model.eval()
    item_vecs = model.item_vectors()
    out = np.zeros((len(seqs), k), dtype=np.int32)
    n_batches = (len(seqs) + batch - 1) // batch
    start = time.time()

    for b, i in enumerate(range(0, len(seqs), batch)):
        sv = model.encode(torch.from_numpy(seqs[i:i + batch]).to(device))
        scores = (sv @ item_vecs.T).cpu()
        scores[:, 0] = -float("inf")  # never recommend padding
        out[i:i + batch] = scores.topk(k).indices.numpy() - 1
        if log_every and (b + 1) % log_every == 0:
            elapsed = time.time() - start
            left = elapsed / (b + 1) * (n_batches - b - 1)
            print(f"batch {b + 1}/{n_batches} | {elapsed / 60:.1f} min | ~{left / 60:.0f} min left")
    return out
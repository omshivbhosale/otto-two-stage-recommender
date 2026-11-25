import numpy as np
import pandas as pd
import torch

from src.model import SessionGRU
from src.retrieval import build_sequences, search


def test_build_sequences_format():
    events = pd.DataFrame({
        "session": [2, 1, 1, 1, 1],
        "aid":     [7, 3, 5, 5, 9],
        "ts":      [1, 4, 2, 1, 3],   # rows out of time order
    })
    sessions, seqs = build_sequences(events, seq_len=4)

    assert list(sessions) == [1, 2]
    # session 1 in time order: 5 (t1), 5 (t2), 9 (t3), 3 (t4)
    # back-to-back repeat removed -> 5, 9, 3 -> +1 -> 6, 10, 4
    assert list(seqs[0]) == [0, 6, 10, 4]
    assert list(seqs[1]) == [0, 0, 0, 8]


def test_build_sequences_keeps_returns_to_an_item():
    events = pd.DataFrame({"session": [1, 1, 1], "aid": [5, 9, 5], "ts": [1, 2, 3]})
    _, seqs = build_sequences(events, seq_len=3)
    assert list(seqs[0]) == [6, 10, 6]  # 5 -> 9 -> 5: not back-to-back, so both 5s stay


def test_build_sequences_keeps_last_items():
    events = pd.DataFrame({"session": [1] * 6, "aid": [10, 11, 12, 13, 14, 15], "ts": range(6)})
    _, seqs = build_sequences(events, seq_len=3)
    assert list(seqs[0]) == [14, 15, 16]  # last 3 items (13, 14, 15) shifted by +1


def test_search_returns_real_items():
    torch.manual_seed(0)
    n_items = 5
    model = SessionGRU(n_items, dim=8)
    seqs = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)

    out = search(model, seqs, k=n_items, batch=1, log_every=0)

    assert out.shape == (2, n_items)
    for row in out:
        # every real item exactly once, never the padding item (-1 after the shift)
        assert sorted(row) == list(range(n_items))
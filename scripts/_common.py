"""Paths and helpers shared by the pipeline scripts."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"

N_ITEMS = 1_855_603  # OTTO item IDs run 0..1,855,602; the GRU shifts them by +1 so 0 can mean padding
TRAINING_SESSIONS = DATA / "rank_training_sessions.npy"

# Saved feature tables, per system version and week.
FEATURE_DIRS = {
    ("v1", "rank"): "rank_features_gru",
    ("v1", "test"): "test_features",
    ("v2", "rank"): "rank_features_v2",
    ("v2", "test"): "test_features_v2",
}


def ranker_path(version: str, target_col: str) -> Path:
    """v1 keeps the original file names (ranker_y_click.txt, ...)."""
    name = f"ranker_{target_col}.txt" if version == "v1" else f"ranker_{version}_{target_col}.txt"
    return MODELS / name


def training_sessions() -> np.ndarray:
    """The ranker-week sessions the rankers train on (chosen once by scripts.train_rankers).

    Every other ranker-week session is held out for choosing between systems.
    """
    return np.load(TRAINING_SESSIONS)


def read_sessions(path: Path, sessions: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Read only the given sessions from a (possibly partitioned) Parquet folder, sorted by session."""
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    session_type = dataset.schema.field("session").type
    table = dataset.to_table(
        columns=columns,
        filter=ds.field("session").isin(pa.array(sessions).cast(session_type)),
    )
    # rankers need each session's rows next to each other
    return table.to_pandas().sort_values("session", kind="stable").reset_index(drop=True)


def add_sim_all(d: pd.DataFrame, prefix: str) -> None:
    """GRU similarity between each session and each candidate, from saved vectors (IDs shifted +1)."""
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

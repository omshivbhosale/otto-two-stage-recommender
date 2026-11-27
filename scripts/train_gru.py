"""Train the GRU session model for one week (PyTorch, Apple GPU if available).

  python -m scripts.train_gru --week rank    # trained on weeks 1-2, used for the ranker week
  python -m scripts.train_gru --week test    # trained on weeks 1-3, used for the test week

Each model only sees events before its week's cutoff. Sessions become their last
20 items (back-to-back repeats removed); the model learns to predict the next item,
with the other items in the batch as negatives.
"""
import argparse
import json
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from scripts._common import DATA, MODELS, N_ITEMS, REPORTS
from src.model import SessionGRU, next_item_loss
from src.retrieval import build_sequences

TRAIN_EVENTS = {"rank": "rank_train", "test": "split_train"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--week", choices=TRAIN_EVENTS, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    start = time.time()

    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    events = pq.read_table(DATA / TRAIN_EVENTS[args.week], columns=["session", "aid", "ts"]).to_pandas()
    _, seqs = build_sequences(events)
    del events
    seqs = seqs[seqs[:, -2] > 0]  # at least 2 items, so there is a "next item" to learn
    X = torch.from_numpy(seqs)
    print(f"device: {device} | training sequences: {tuple(X.shape)}", flush=True)

    model = SessionGRU(N_ITEMS, dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = len(X) // args.batch
    history = []

    model.train()
    for epoch in range(args.epochs):
        perm = torch.randperm(len(X))
        losses = []
        for step in range(steps):
            seq = X[perm[step * args.batch:(step + 1) * args.batch]].to(device)
            opt.zero_grad()
            loss = next_item_loss(model, seq)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if (step + 1) % 500 == 0:
                print(f"epoch {epoch + 1} | step {step + 1}/{steps} | loss {np.mean(losses[-500:]):.3f} | "
                      f"{(time.time() - start) / 60:.1f} min", flush=True)
        history.append({"epoch": epoch + 1, "last_500_steps_loss": float(np.mean(losses[-500:]))})

    MODELS.mkdir(exist_ok=True)
    out = MODELS / f"gru_{args.week}.pt"
    torch.save(model.state_dict(), out)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"gru_{args.week}.json").write_text(json.dumps(
        {"week": args.week, "sequences": int(len(X)), "batch": args.batch, "lr": args.lr, "history": history},
        indent=2,
    ))
    print(f"saved {out.relative_to(MODELS.parent)} | {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()

"""GRU session model for candidate retrieval."""
import torch
import torch.nn.functional as F
from torch import nn


class SessionGRU(nn.Module):
    """Reads a session's items in order and outputs a vector at each step.

    Item IDs are shifted by +1 so that 0 means padding.
    """

    def __init__(self, n_items: int, dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.gru = nn.GRU(dim, dim, batch_first=True)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (batch, length) item IDs -> (batch, length, dim) session vectors."""
        out, _ = self.gru(self.emb(seq))
        return out

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """One normalized vector per session: the state after the last item."""
        return F.normalize(self.forward(seq)[:, -1], dim=-1)

    def item_vectors(self) -> torch.Tensor:
        """Normalized embedding for every item (row 0 is padding)."""
        return F.normalize(self.emb.weight, dim=-1)


def next_item_loss(
    model: SessionGRU, seq: torch.Tensor, max_pairs: int = 4096, temperature: float = 0.1
) -> torch.Tensor:
    """Train to predict the next item at every position, using in-batch negatives.

    For each (items so far -> next item) pair, the true next item should score
    higher than the next items of the other pairs in the batch.
    """
    states = model(seq[:, :-1])          # (B, L-1, dim)
    targets = seq[:, 1:]                 # (B, L-1)
    valid = (targets > 0) & (seq[:, :-1] > 0)

    s = F.normalize(states[valid], dim=-1)
    t = targets[valid]
    if len(t) > max_pairs:
        keep = torch.randperm(len(t), device=t.device)[:max_pairs]
        s, t = s[keep], t[keep]

    t_vecs = F.normalize(model.emb(t), dim=-1)
    logits = s @ t_vecs.T / temperature  # (pairs, pairs)

    # if the same item is a target twice, don't punish matching the duplicate
    same = t.unsqueeze(0) == t.unsqueeze(1)
    logits = logits.masked_fill(same & ~torch.eye(len(t), dtype=torch.bool, device=t.device), -1e9)

    return F.cross_entropy(logits, torch.arange(len(t), device=t.device))
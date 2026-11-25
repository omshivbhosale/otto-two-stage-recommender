import torch

from src.model import SessionGRU, next_item_loss

N_ITEMS = 10


def toy_batch():
    # 10 sessions that always step forward: 1->2->3->..., wrapping after 10
    return torch.tensor([[(i + k) % N_ITEMS + 1 for k in range(5)] for i in range(10)])


def test_shapes_and_padding():
    torch.manual_seed(0)
    model = SessionGRU(N_ITEMS, dim=8)
    seq = torch.tensor([[0, 0, 3, 4, 5], [1, 2, 3, 4, 5]])

    assert model(seq).shape == (2, 5, 8)
    assert model.encode(seq).shape == (2, 8)
    assert model.item_vectors().shape == (N_ITEMS + 1, 8)
    # padding row stays all zeros
    assert torch.all(model.emb.weight[0] == 0)
    # session vectors are normalized to length 1
    assert torch.allclose(model.encode(seq).norm(dim=-1), torch.ones(2), atol=1e-5)


def test_loss_is_a_sensible_number():
    torch.manual_seed(0)
    model = SessionGRU(N_ITEMS, dim=8)
    loss = next_item_loss(model, toy_batch())
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_model_learns_a_simple_pattern():
    torch.manual_seed(0)
    model = SessionGRU(N_ITEMS, dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    seq = toy_batch()

    start = next_item_loss(model, seq).item()
    for _ in range(200):
        opt.zero_grad()
        loss = next_item_loss(model, seq)
        loss.backward()
        opt.step()

    assert loss.item() < start * 0.5
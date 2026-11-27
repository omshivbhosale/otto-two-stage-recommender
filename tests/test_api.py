import pytest
from fastapi.testclient import TestClient

import src.api as api


class FakeRecommender:
    """Stands in for the real model: returns fixed lists and remembers its input."""

    def __init__(self, artifacts, models):
        self.last_events = None

    def recommend(self, events, k=20):
        self.last_events = events
        return {"clicks": list(range(k)), "carts": list(range(k, 2 * k)), "orders": list(range(2 * k, 3 * k))}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "Recommender", FakeRecommender)
    with TestClient(api.app) as c:  # "with" runs the startup step that loads the recommender
        yield c


GOOD_EVENT = {"aid": 1517085, "ts": 1659304800, "type": 0}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_recommend_returns_a_list_per_event_type(client):
    r = client.post("/recommend", json={"events": [GOOD_EVENT], "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["clicks"] == [0, 1, 2]
    assert body["carts"] == [3, 4, 5]
    assert body["orders"] == [6, 7, 8]
    assert body["latency_ms"] >= 0


def test_events_reach_the_recommender(client):
    client.post("/recommend", json={"events": [GOOD_EVENT]})
    assert api.state["rec"].last_events == [GOOD_EVENT]


def test_default_k_is_20(client):
    body = client.post("/recommend", json={"events": [GOOD_EVENT]}).json()
    assert len(body["clicks"]) == len(body["carts"]) == len(body["orders"]) == 20


def test_empty_session_is_allowed(client):
    r = client.post("/recommend", json={"events": []})
    assert r.status_code == 200


@pytest.mark.parametrize("bad_body", [
    {"events": [{"aid": -1, "ts": 1, "type": 0}]},          # negative item ID
    {"events": [{"aid": 1_855_603, "ts": 1, "type": 0}]},   # item ID past the catalog
    {"events": [{"aid": 1, "ts": 1, "type": 3}]},           # unknown event type
    {"events": [{"aid": 1, "type": 0}]},                    # missing timestamp
    {"events": [GOOD_EVENT], "k": 0},                       # k too small
    {"events": [GOOD_EVENT], "k": 51},                      # k too large
    {"events": [GOOD_EVENT] * 501},                         # session too long
])
def test_bad_input_is_rejected(client, bad_body):
    r = client.post("/recommend", json=bad_body)
    assert r.status_code == 422

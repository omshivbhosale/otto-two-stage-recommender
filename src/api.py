"""FastAPI service: POST a session's events, get back recommended item IDs."""
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.serve import N_AIDS, Recommender

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts"))
MODELS = Path(os.environ.get("MODELS_DIR", ROOT / "models"))

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the models once at startup, not on every request
    state["rec"] = Recommender(ARTIFACTS, MODELS)
    yield
    state.clear()


app = FastAPI(title="Two-stage e-commerce recommender", lifespan=lifespan)


class Event(BaseModel):
    aid: int = Field(ge=0, lt=N_AIDS, description="item ID")
    ts: int = Field(ge=0, description="Unix timestamp in seconds")
    type: int = Field(ge=0, le=2, description="0 = click, 1 = cart, 2 = order")


class RecommendRequest(BaseModel):
    events: list[Event] = Field(default_factory=list, max_length=500)
    k: int = Field(default=20, ge=1, le=50)


class RecommendResponse(BaseModel):
    items: list[int]
    latency_ms: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "rec" in state}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    start = time.perf_counter()
    items = state["rec"].recommend([e.model_dump() for e in req.events], k=req.k)
    return RecommendResponse(items=items, latency_ms=round((time.perf_counter() - start) * 1000, 2))
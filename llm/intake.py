"""The intake: passengers' reviews come in here, wait in the queue, and are handed to the trainer in batches.

    uvicorn llm.intake:app --host 0.0.0.0 --port 8001     (Docker: the `intake` service)

    POST /reviews       one passenger review with the passenger's own score, recommendation and sub-ratings
                        (the simulator, simulator/replay.py, posts the local dataset's reviews here in date order)
    POST /queue/claim   {"n": 2000, "replay": 500} -> a batch id, the newest pending reviews, some already-trained ones
    POST /queue/done    {"batch_id": ..., "ok": true}   ok: trained; false: back to pending
    GET  /queue/stats   counts per status (numbers only)

A background loop sends every new review to the complaint-box API (SCORE_URL, POST /score) with its true labels,
oldest first, a few at a time: that is the live traffic the dashboard and the drift check measure. Reviews posted
with "live": false (the first model's training reviews) are queued but not scored.
It also puts back claims older than queue_db.STALE_HOURS (a trainer pod that died), and with AUTO_TRAIN=1 starts a
trainer pod on RunPod (llm/runpod_pod.py) when TRAIN_MIN_PENDING reviews wait.
"""

import json
import logging
import os
import threading
import time
import urllib.request
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict, Field

from llm import queue_db
from llm.reviews import COMPLAINT_NAMES, complaints_from

DATABASE_URL = os.environ.get("DATABASE_URL")
SCORE_URL = os.environ.get("SCORE_URL", "")  # e.g. http://api:8000/score; empty: no live scoring
SCORE_SECONDS = float(os.environ.get("SCORE_SECONDS", "5"))
SCORE_BATCH = int(os.environ.get("SCORE_BATCH", "5"))
AUTO_TRAIN = os.environ.get("AUTO_TRAIN", "0") == "1"
TRAIN_MIN_PENDING = int(os.environ.get("TRAIN_MIN_PENDING", "500"))
TRAIN_EVERY_HOURS = float(os.environ.get("TRAIN_EVERY_HOURS", "6"))  # at most one auto-started pod per this long

log = logging.getLogger("uvicorn.error")
# autocommit: each statement commits at once, so claims never wait on the scoring loop
db = (ConnectionPool(DATABASE_URL, min_size=1, max_size=4, open=False, kwargs={"autocommit": True})
      if DATABASE_URL else None)
state = {"scored": 0, "score_errors": 0, "last_error": None, "last_pod": None, "last_pod_at": 0.0}


class Ratings(BaseModel):
    """The passenger's sub-ratings, 1-5; 0 or missing: not rated."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    seat: int | None = Field(None, ge=0, le=5)
    service: int | None = Field(None, ge=0, le=5)
    food: int | None = Field(None, ge=0, le=5)
    entertainment: int | None = Field(None, ge=0, le=5)
    ground_service: int | None = Field(None, ge=0, le=5, alias="ground service")
    wifi: int | None = Field(None, ge=0, le=5)
    value: int | None = Field(None, ge=0, le=5)

    def by_name(self) -> dict[str, int]:
        values = self.model_dump(by_alias=True)
        return {name: values[name] for name in COMPLAINT_NAMES if values.get(name)}


class ReviewIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_id: str | None = Field(None, max_length=100, description="the sender's id; a repeat is ignored")
    airline: str = Field(min_length=1, max_length=200)
    cabin: str = Field("", max_length=100)
    travel_type: str = Field("", max_length=100)
    title: str = Field("", max_length=500)
    review: str = Field(min_length=1, max_length=50_000)
    flown: date | None = None
    score: int = Field(ge=1, le=10, description="the passenger's own overall score")
    recommended: Literal["yes", "no"]
    ratings: Ratings = Ratings()
    source: Literal["passenger", "simulator"] = "passenger"
    live: bool = Field(True, description="false: queue it for training only, do not send it to the model as traffic")


def queue_row(req: ReviewIn) -> dict:
    return {"id": req.review_id or f"{req.source}:{uuid.uuid4().hex}", "source": req.source,
            "airline": req.airline, "cabin": req.cabin, "travel_type": req.travel_type, "title": req.title,
            "review": req.review, "flown": req.flown, "score": req.score, "recommended": req.recommended,
            "complaints": complaints_from(req.ratings.by_name()),
            "scored_at": None if req.live else datetime.now(timezone.utc)}  # training-only: never live traffic


def score_body(row: dict) -> dict:
    return {"airline": row["airline"], "cabin": row["cabin"] or "", "travel_type": row["travel_type"] or "",
            "title": row["title"] or "", "review": row["review"][:50_000],
            "true_score": row["score"], "true_recommended": row["recommended"]}


def score_some(conn, n: int) -> int:
    """Sends up to n unscored reviews to the API with their true labels. Returns how many were answered."""
    done = 0
    for row in queue_db.to_score(conn, n):
        req = urllib.request.Request(SCORE_URL, data=json.dumps(score_body(row)).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=150) as resp:
                resp.read()
        except Exception as err:  # the API is loading, busy or has no champion yet: try again next round
            state["score_errors"] += 1
            state["last_error"] = str(err)
            break
        queue_db.mark_scored(conn, row["id"])
        done += 1
    state["scored"] += done
    return done


def maybe_train(conn) -> None:
    if not AUTO_TRAIN:
        return
    s = queue_db.stats(conn)
    if s["pending"] < TRAIN_MIN_PENDING or s["claimed"] or time.time() - state["last_pod_at"] < TRAIN_EVERY_HOURS * 3600:
        return
    from llm import runpod_pod
    pod = runpod_pod.launch(log=log.info)
    state["last_pod"], state["last_pod_at"] = pod["id"], time.time()
    log.info("Started trainer pod %s for %d pending reviews", pod["id"], s["pending"])


def background() -> None:
    last_housekeeping = 0.0
    while True:
        try:
            with db.connection() as conn:
                if time.time() - last_housekeeping > 300:
                    released = queue_db.release_stale(conn)
                    if released:
                        log.warning("Put %d reviews of a dead batch back to pending", released)
                    maybe_train(conn)
                    last_housekeeping = time.time()
                if SCORE_URL:
                    score_some(conn, SCORE_BATCH)
        except Exception as err:
            state["last_error"] = str(err)
            log.warning("Intake loop: %s", err)
        time.sleep(SCORE_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if db is None:
        raise RuntimeError("DATABASE_URL is not set: the queue lives in PostgreSQL")
    db.open()
    with db.connection() as conn:
        queue_db.create(conn)
    threading.Thread(target=background, daemon=True).start()
    yield
    db.close()


app = FastAPI(title="SkySat review intake and queue", version="1.0", lifespan=lifespan)


@app.post("/reviews")
def add_review(req: ReviewIn):
    row = queue_row(req)
    with db.connection() as conn:
        added = queue_db.insert(conn, [row])
    return {"id": row["id"], "queued": bool(added), "complaints": row["complaints"]}


class ClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = Field(2000, ge=1, le=20_000)
    replay: int = Field(0, ge=0, le=10_000)


class DoneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_id: str = Field(min_length=1, max_length=100)
    ok: bool


@app.post("/queue/claim")
def claim(req: ClaimRequest):
    batch_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    with db.connection() as conn:
        batch = queue_db.claim(conn, batch_id, req.n, req.replay)
    log.info("Batch %s: %d new reviews, %d replayed", batch_id, len(batch["reviews"]), len(batch["replay"]))
    return batch


@app.post("/queue/done")
def done(req: DoneRequest):
    with db.connection() as conn:
        changed = queue_db.finish(conn, req.batch_id, req.ok)
    if not changed:
        raise HTTPException(status_code=404, detail="No claimed reviews for that batch (finished, or released as stale)")
    log.info("Batch %s %s: %d reviews", req.batch_id, "trained" if req.ok else "put back", changed)
    return {"batch_id": req.batch_id, "status": "trained" if req.ok else "pending", "reviews": changed}


@app.get("/queue/stats")
def stats():
    with db.connection() as conn:
        s = queue_db.stats(conn)
    return s | {"intake": {k: v for k, v in state.items() if k != "last_pod_at"}}


@app.get("/health")
def health():
    with db.connection(timeout=5) as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}

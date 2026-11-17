"""The intake, the queue's SQL and the trainer together, on a real PostgreSQL (TEST_DATABASE_URL names a *_test
database: CI starts one). No HTTP server and no GPU: trainer.call goes straight to the intake's handler functions and
trainer.run, the GPU training, is a stub that records the frames it was given."""

import os

import pytest
from fastapi import HTTPException

from llm import intake, queue_db, trainer
from llm.intake import ClaimRequest, DoneRequest, ReviewIn


@pytest.fixture
def pool(monkeypatch):
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set (CI starts a PostgreSQL)")
    # The tests drop the queue schema: never point them at a database that holds a real queue
    assert url.rstrip("/").rsplit("/", 1)[-1].endswith("_test"), "TEST_DATABASE_URL must name a *_test database"
    psycopg = pytest.importorskip("psycopg")
    from psycopg_pool import ConnectionPool
    with psycopg.connect(url, autocommit=True) as c:
        c.execute("DROP SCHEMA IF EXISTS queue CASCADE")
        queue_db.create(c)
    test_pool = ConnectionPool(url, min_size=1, max_size=4, open=False, kwargs={"autocommit": True})
    test_pool.open()
    monkeypatch.setattr(intake, "db", test_pool)
    yield test_pool
    test_pool.close()
    with psycopg.connect(url, autocommit=True) as c:
        c.execute("DROP SCHEMA IF EXISTS queue CASCADE")


def post(i, prefix="p", live=False, **changes):
    """A made-up review with a fixed id and a text of its own, posted through the intake's POST /reviews handler."""
    fields = {"review_id": f"{prefix}:{i}", "airline": "Imaginary Air", "cabin": "Economy Class",
              "travel_type": "Solo Leisure", "title": "Cold", "review": f"The lasagne was cold on flight {prefix}{i}.",
              "flown": "2019-03-01", "score": 1 + i % 10, "recommended": "no",
              "ratings": {"food": 1, "seat": 4, "ground service": 1}, "live": live} | changes
    return intake.add_review(ReviewIn(**fields))


def post_many(n, prefix="p", **changes):
    for i in range(n):
        post(i, prefix, **changes)
    return {f"{prefix}:{i}" for i in range(n)}


def fake_call(url, path, body=None, timeout=120):
    """What trainer.call does over HTTP, against the intake's own handlers: JSON-able dicts, errors for HTTP errors."""
    try:
        if path == "/queue/stats":
            return intake.stats()
        if path == "/queue/claim":
            return intake.claim(ClaimRequest(**body))
        if path == "/queue/done":
            return intake.done(DoneRequest(**body))
    except HTTPException as err:
        raise RuntimeError(f"HTTP {err.status_code}: {err.detail}") from err
    raise AssertionError(f"unexpected request {path}")


class Training:
    """A stand-in for trainer.run: records the frames it was given, returns what it was told to."""

    def __init__(self, registered=True, crash=None):
        self.registered, self.crash, self.calls = registered, crash, []

    def __call__(self, train_rows, gate_rows, args, params=None):
        self.calls.append({"train": set(train_rows["id"]), "gate": set(gate_rows["id"]), "params": params})
        if self.crash:
            raise self.crash
        return {"metrics": {}, "registered": self.registered, "version": "7" if self.registered else None,
                "promoted": self.registered, "champion_combined": None}


def run_trainer(monkeypatch, training, *flags):
    monkeypatch.setattr(trainer, "call", fake_call)
    monkeypatch.setattr(trainer, "run", training)
    trainer.main(["--queue-url", "http://queue", "--min-new", "20", "--n-gate-min", "5", *flags])


def counts():
    s = intake.stats()
    return {k: s[k] for k in ("pending", "claimed", "trained", "total")}


def test_a_registered_version_marks_the_batch_trained(pool, monkeypatch):
    posted = post_many(40)
    training = Training(registered=True)
    run_trainer(monkeypatch, training)
    assert counts() == {"pending": 0, "claimed": 0, "trained": 40, "total": 40}
    (call,) = training.calls
    assert call["gate"] and call["train"] and not call["gate"] & call["train"]
    assert call["gate"] | call["train"] == posted


def test_a_failed_gate_gives_the_batch_back(pool, monkeypatch):
    post_many(40)
    training = Training(registered=False)
    run_trainer(monkeypatch, training)
    assert len(training.calls) == 1
    assert counts() == {"pending": 40, "claimed": 0, "trained": 0, "total": 40}


def test_a_crash_gives_the_batch_back(pool, monkeypatch):
    post_many(40)
    with pytest.raises(RuntimeError, match="out of memory"):
        run_trainer(monkeypatch, Training(crash=RuntimeError("out of memory")))
    assert counts() == {"pending": 40, "claimed": 0, "trained": 0, "total": 40}


def test_too_few_new_reviews_trains_nothing(pool, monkeypatch):
    post_many(5)
    training = Training()
    run_trainer(monkeypatch, training)
    assert training.calls == []
    assert counts() == {"pending": 5, "claimed": 0, "trained": 0, "total": 5}


def test_the_second_run_trains_on_the_new_reviews_and_replays_old_ones(pool, monkeypatch):
    old = post_many(40, "old")
    run_trainer(monkeypatch, Training(registered=True))
    assert counts()["trained"] == 40
    new = post_many(30, "new")
    second = Training(registered=True)
    run_trainer(monkeypatch, second, "--replay", "10")
    (call,) = second.calls
    assert call["gate"] and call["gate"] <= new  # the gate is drawn from the new reviews only
    assert call["train"] & old and call["train"] <= old | new  # some old reviews are replayed
    assert not call["gate"] & call["train"]
    assert counts() == {"pending": 0, "claimed": 0, "trained": 70, "total": 70}


def test_the_same_review_id_posted_twice_keeps_one_row(pool):
    assert post(1)["queued"] is True
    total = counts()["total"]
    assert post(1)["queued"] is False
    assert counts()["total"] == total == 1


def test_a_claimed_review_is_not_handed_out_again(pool):
    post_many(30)
    first = intake.claim(ClaimRequest(n=10))
    second = intake.claim(ClaimRequest(n=10))
    ids = lambda batch: {r["id"] for r in batch["reviews"]}
    assert len(ids(first)) == len(ids(second)) == 10 and not ids(first) & ids(second)
    assert counts()["claimed"] == 20

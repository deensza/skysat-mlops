"""The intake and the trainer without a GPU: the queue's SQL (llm/queue_db.py, against a real PostgreSQL when
TEST_DATABASE_URL names a *_test database: CI starts one), the intake's request model (llm/intake.py), the trainer's
batch preparation (llm/trainer.py) and the RunPod request (llm/runpod_pod.py)."""

import os
import uuid

import pytest

from llm import intake, queue_db, runpod_pod, trainer
from llm.intake import ReviewIn, queue_row, score_body
from llm.trainer import prepare


def review(**changes) -> dict:
    return {"airline": "Imaginary Air", "cabin": "Economy Class", "travel_type": "Solo Leisure", "title": "Cold",
            "review": "The lasagne was cold.", "flown": "2019-03-01", "score": 3, "recommended": "no",
            "ratings": {"food": 1, "seat": 4, "ground service": 1}} | changes


def test_intake_derives_the_complaint_from_the_ratings():
    row = queue_row(ReviewIn(**review()))
    assert row["complaints"] == ["food", "ground service"] and row["source"] == "passenger"
    assert row["id"].startswith("passenger:")
    assert queue_row(ReviewIn(**review(ratings={})))["complaints"] == ["none"]
    assert queue_row(ReviewIn(**review(review_id="kaggle128k:7")))["id"] == "kaggle128k:7"
    assert row["scored_at"] is None and queue_row(ReviewIn(**review(live=False)))["scored_at"] is not None


def test_intake_rejects_bad_reviews():
    for bad in [{"score": 0}, {"score": 11}, {"recommended": "maybe"}, {"review": ""}, {"ratings": {"food": 6}},
                {"ratings": {"legroom": 1}}, {"extra": 1}]:
        with pytest.raises(ValueError):
            ReviewIn(**review(**bad))


def test_live_traffic_body_carries_the_truth_never_the_id():
    row = queue_row(ReviewIn(**review()))
    body = score_body(row)
    assert body["true_score"] == 3 and body["true_recommended"] == "no" and "id" not in body


def batch(n_new=100, n_replay=20):
    row = lambda i, s: {"id": f"x:{i}", "airline": "A", "cabin": "Economy Class", "travel_type": "Business",
                        "title": "T", "review": f"text {i}", "flown": "2019-01-01", "score": s,
                        "recommended": "yes" if s >= 6 else "no", "complaints": ["none"]}
    return {"batch_id": "b1", "reviews": [row(i, 1 + i % 10) for i in range(n_new)],
            "replay": [row(1000 + i, 9) for i in range(n_replay)]}


def test_trainer_gates_only_on_new_reviews_never_trained_on():
    train_rows, gate_rows = prepare(batch(), n_gate_max=200, seed=0)
    assert len(gate_rows) == 10 and len(train_rows) == 90 + 20
    assert not set(train_rows["id"]) & set(gate_rows["id"])
    assert all(int(i.split(":")[1]) < 1000 for i in gate_rows["id"])  # no replayed row in the gate
    assert {"prompt", "completion"} <= set(train_rows.columns)


def test_trainer_never_trains_on_a_gate_text():
    b = batch(n_new=200, n_replay=0)
    for r in b["reviews"]:
        r["review"] = f"Great flight {int(r['id'].split(':')[1]) % 50}"  # only 50 distinct texts
    b["replay"] = [dict(r, id=f"r:{i}") for i, r in enumerate(b["reviews"][:20])]
    train_rows, gate_rows = prepare(b, n_gate_max=200, seed=0)
    assert gate_rows["Review"].is_unique and not set(train_rows["Review"]) & set(gate_rows["Review"])


def test_trainer_caps_the_gate():
    assert len(prepare(batch(n_new=1000, n_replay=0), n_gate_max=30, seed=0)[1]) == 30


def test_trainer_keeps_a_gate_big_enough_to_mean_something():
    sizes = lambda n: [len(f) for f in prepare(batch(n_new=n, n_replay=0), n_gate_max=200, seed=0, n_gate_min=100)]
    assert sizes(360) == [260, 100]    # the demo's drift batch: 10% would be 36 gate reviews
    assert sizes(2000) == [1800, 200]  # a full batch: 10%, as before
    assert sizes(150) == [75, 75]      # never more than half of the batch


def test_runpod_request_is_cheap_safe_and_self_deleting(monkeypatch):
    monkeypatch.setenv("QUEUE_PASSWORD", "secret")
    body = runpod_pod.pod_body(runpod_pod.GPUS[0], "main", "--n 300", 60)
    assert body["gpu"] == {"id": "NVIDIA GeForce RTX 3070", "count": 1, "minCudaVersion": "12.8"}
    assert body["cloud"] == "COMMUNITY" and body["env"]["MAX_MINUTES"] == "60"
    assert body["env"]["TRAINER_ARGS"] == "--n 300" and body["env"]["QUEUE_PASSWORD"] == "secret"
    script = body["cmd"][0]
    assert script.count("delete_pod") >= 3 and "sleep $((MAX_MINUTES * 60))" in script
    assert "curl -sf" in script and "POD DELETE FAILED: delete pod $RUNPOD_POD_ID by hand" in script
    assert "RUNPOD_API_KEY" not in body["env"]  # the pod uses the key RunPod gives it, never the account's


def test_runpod_launch_needs_the_queue_login(monkeypatch):
    for k in ("QUEUE_URL", "QUEUE_USER", "QUEUE_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit):
        runpod_pod.launch()


def run_trainer(monkeypatch, n_new, result, *flags):
    """trainer.main() with the queue and the training faked; returns (the ok posted to /queue/done, run calls)."""
    posts, runs = [], []

    def fake_call(url, path, body=None, timeout=120):
        if path == "/queue/done":
            posts.append(body["ok"])
        return batch(n_new=n_new, n_replay=0) if path == "/queue/claim" else {"pending": 0}

    def fake_run(*a, **k):
        runs.append(1)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(trainer, "call", fake_call)
    monkeypatch.setattr(trainer, "run", fake_run)
    monkeypatch.setattr(trainer, "prepare", lambda *a: ([], []))
    trainer.main(["--queue-url", "http://q", "--min-new", "50", *flags])
    return posts, runs


REGISTERED = {"registered": True, "version": "3", "promoted": True}
NOT_REGISTERED = {"registered": False, "version": None, "promoted": False}


def test_trainer_marks_trained_only_when_a_version_was_registered(monkeypatch):
    assert run_trainer(monkeypatch, 100, REGISTERED)[0] == [True]
    assert run_trainer(monkeypatch, 100, NOT_REGISTERED)[0] == [False]  # the gate failed
    assert run_trainer(monkeypatch, 100, NOT_REGISTERED, "--no-mlflow")[0] == [False]
    assert run_trainer(monkeypatch, 100, REGISTERED, "--keep-pending")[0] == [False]


def test_trainer_gives_the_batch_back_when_training_raises(monkeypatch):
    posts = []
    monkeypatch.setattr(trainer, "call", lambda url, path, body=None, timeout=120:
                        posts.append(body["ok"]) or {} if path == "/queue/done" else
                        batch(n_new=100, n_replay=0) if path == "/queue/claim" else {})
    monkeypatch.setattr(trainer, "prepare", lambda *a: ([], []))
    monkeypatch.setattr(trainer, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("out of memory")))
    with pytest.raises(RuntimeError, match="out of memory"):
        trainer.main(["--queue-url", "http://q", "--min-new", "50"])
    assert posts == [False]


def test_trainer_with_too_few_new_reviews_trains_nothing(monkeypatch):
    posts, runs = run_trainer(monkeypatch, 10, REGISTERED)
    assert posts == [False] and runs == []


@pytest.fixture
def auto(monkeypatch):
    """maybe_train with auto-training on, enough pending reviews and a fresh state."""
    monkeypatch.setattr(intake, "auto_train", True)
    monkeypatch.setattr(intake, "state", {"scored": 0, "score_errors": 0, "last_error": None, "last_pod": None,
                                          "last_pod_at": 0.0, "last_train_error": None})
    monkeypatch.setattr(queue_db, "stats", lambda conn: {"pending": 10_000, "claimed": 0})
    launches = []
    monkeypatch.setattr(runpod_pod, "launch", lambda log=print: launches.append(1) or {"id": "pod1"})
    return launches


@pytest.mark.parametrize("failure", [SystemExit("Set RUNPOD_API_KEY"), RuntimeError("No GPU had capacity")])
def test_auto_train_failure_is_recorded_and_not_retried_at_once(monkeypatch, auto, failure):
    def fail(log=print):
        auto.append(1)
        raise failure
    monkeypatch.setattr(runpod_pod, "launch", fail)
    intake.maybe_train(None)  # must not raise
    assert intake.state["last_train_error"] == str(failure) and auto == [1]
    intake.maybe_train(None)
    assert auto == [1]  # one attempt per TRAIN_EVERY_HOURS


def test_auto_train_success_records_the_pod_and_clears_the_error(auto):
    intake.state["last_train_error"] = "old"
    intake.maybe_train(None)
    assert auto == [1] and intake.state["last_pod"] == "pod1" and intake.state["last_train_error"] is None


def test_auto_train_waits_for_enough_pending_reviews(monkeypatch, auto):
    monkeypatch.setattr(queue_db, "stats", lambda conn: {"pending": 1, "claimed": 0})
    intake.maybe_train(None)
    assert auto == []


def test_runpod_missing_settings_names_the_api_key_too(monkeypatch):
    for k in ("RUNPOD_API_KEY", "QUEUE_URL", "QUEUE_USER", "QUEUE_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    assert runpod_pod.missing_settings() == ["RUNPOD_API_KEY", "QUEUE_URL", "QUEUE_USER", "QUEUE_PASSWORD"]


def test_runpod_outcome_is_read_from_the_pod_log():
    log = ["c614cb7 Merge", "Result: registered=True version=3 promoted=True", "trainer exit code 0", "deleting pod x"]
    assert runpod_pod.outcome(log) == {"exit_code": 0, "registered": True, "version": "3", "promoted": True}
    failed = runpod_pod.outcome(["Result: registered=False version=None promoted=False", "trainer exit code 0"])
    assert failed == {"exit_code": 0, "registered": False, "version": None, "promoted": False}
    assert runpod_pod.outcome(["Traceback", "trainer exit code 1"])["exit_code"] == 1
    assert runpod_pod.outcome(["still pulling the image"])["exit_code"] is None  # never seen: the launcher fails


class FakeRunpod:
    """Stands in for runpod_pod.request: a pod that is there for `alive` reads, then gone (404)."""

    def __init__(self, alive, pods=()):
        self.alive, self.calls, self.pods = alive, [], list(pods)

    def __call__(self, method, path, body=None, timeout=60):
        self.calls.append((method, path))
        if (method, path) == ("GET", "/pods"):
            return {"pods": self.pods}
        if method == "GET":
            self.alive -= 1
            if self.alive < 0:
                raise runpod_pod.urllib.error.HTTPError(path, 404, "gone", None, None)
        return {}


@pytest.fixture
def no_waiting(monkeypatch):
    monkeypatch.setattr(runpod_pod.time, "sleep", lambda s: None)
    monkeypatch.setattr(runpod_pod, "_print_logs", lambda pod_id, seen, log: seen.add((1, "trainer exit code 0")))


def test_runpod_follow_does_not_delete_a_pod_that_deleted_itself(monkeypatch, no_waiting):
    fake = FakeRunpod(alive=2)
    monkeypatch.setattr(runpod_pod, "request", fake)
    assert runpod_pod.follow("p1", 90, log=lambda s: None)["exit_code"] == 0
    assert ("DELETE", "/pods/p1") not in fake.calls


def test_runpod_follow_deletes_the_pod_at_the_deadline(monkeypatch, no_waiting):
    fake = FakeRunpod(alive=10**6)
    monkeypatch.setattr(runpod_pod, "request", fake)
    runpod_pod.follow("p1", 0, log=lambda s: None)  # 0 minutes: the deadline has passed at the first look
    assert fake.calls[-1] == ("DELETE", "/pods/p1")


def test_runpod_follow_deletes_the_pod_when_it_is_interrupted(monkeypatch, no_waiting):
    fake = FakeRunpod(alive=10**6)
    monkeypatch.setattr(runpod_pod, "request", fake)
    monkeypatch.setattr(runpod_pod, "_print_logs", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):  # Ctrl-C, or a cancelled workflow
        runpod_pod.follow("p1", 90, log=lambda s: None)
    assert fake.calls[-1] == ("DELETE", "/pods/p1")


def test_runpod_follow_survives_a_network_blip(monkeypatch, no_waiting):
    gone = runpod_pod.urllib.error.HTTPError("x", 404, "gone", None, None)
    answers = iter([OSError("network is unreachable"), {}, gone])

    def flaky(method, path, body=None, timeout=60):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(runpod_pod, "request", flaky)
    assert runpod_pod.follow("p1", 90, log=lambda s: None)["exit_code"] == 0


def test_runpod_follow_deletes_a_pod_that_stopped_but_is_still_listed(monkeypatch, no_waiting):
    calls = []

    def stopped(method, path, body=None, timeout=60):
        calls.append((method, path))
        return {"status": "EXITED"}

    monkeypatch.setattr(runpod_pod, "request", stopped)
    runpod_pod.follow("p1", 90, log=lambda s: None)
    assert calls == [("GET", "/pods/p1"), ("DELETE", "/pods/p1")]  # no waiting for the deadline


def test_runpod_launch_stops_at_once_when_the_key_is_refused(monkeypatch):
    for k in ("QUEUE_URL", "QUEUE_USER", "QUEUE_PASSWORD"):
        monkeypatch.setenv(k, "x")
    tried = []

    def refused(method, path, body=None, timeout=60):
        tried.append(body["gpu"]["id"])
        raise runpod_pod.urllib.error.HTTPError(path, 401, "Unauthorized", None, __import__("io").BytesIO(b"bad key"))

    monkeypatch.setattr(runpod_pod, "request", refused)
    with pytest.raises(RuntimeError, match="refused the API key"):
        runpod_pod.launch(log=lambda s: None)
    assert len(tried) == 1  # not once per GPU type


def test_runpod_launch_sweeps_when_the_answer_to_create_is_lost(monkeypatch, no_waiting):
    for k in ("QUEUE_URL", "QUEUE_USER", "QUEUE_PASSWORD"):
        monkeypatch.setenv(k, "x")
    calls = []

    def lost(method, path, body=None, timeout=60):
        calls.append((method, path))
        if method == "POST":
            raise TimeoutError("timed out")  # RunPod may have made the pod all the same
        return {"pods": [{"id": "a", "name": "skysat-trainer"}]} if path == "/pods" else {}

    monkeypatch.setattr(runpod_pod, "request", lost)
    with pytest.raises(RuntimeError, match="Could not reach RunPod"):
        runpod_pod.launch(log=lambda s: None)
    assert calls == [("POST", "/pods"), ("GET", "/pods"), ("DELETE", "/pods/a")]


def test_runpod_sweep_deletes_only_trainer_pods(monkeypatch, no_waiting):
    fake = FakeRunpod(alive=0, pods=[{"id": "a", "name": "skysat-trainer"}, {"id": "b", "name": "someone-elses-pod"}])
    monkeypatch.setattr(runpod_pod, "request", fake)
    assert runpod_pod.sweep(log=lambda s: None) == 1
    assert ("DELETE", "/pods/a") in fake.calls and ("DELETE", "/pods/b") not in fake.calls


# --- The queue's SQL, on a real PostgreSQL -------------------------------------------------------------------

@pytest.fixture
def conn():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set (CI starts a PostgreSQL)")
    # The tests drop the queue schema: never point them at a database that holds a real queue
    assert url.rstrip("/").rsplit("/", 1)[-1].endswith("_test"), "TEST_DATABASE_URL must name a *_test database"
    psycopg = pytest.importorskip("psycopg")
    with psycopg.connect(url, autocommit=True) as c:
        c.execute("DROP SCHEMA IF EXISTS queue CASCADE")
        queue_db.create(c)
        yield c
        c.execute("DROP SCHEMA IF EXISTS queue CASCADE")


def rows(n, start=0):
    return [queue_row(ReviewIn(**review(review_id=f"t:{i}", review=f"review {i}", score=1 + i % 10)))
            for i in range(start, start + n)]


def test_queue_insert_is_idempotent(conn):
    assert queue_db.insert(conn, rows(10)) == 10
    assert queue_db.insert(conn, rows(12)) == 2  # only the 2 new ones
    s = queue_db.stats(conn)
    assert s["pending"] == 12 and s["unscored"] == 12
    training_only = queue_row(ReviewIn(**review(review_id="t:train", live=False)))
    queue_db.insert(conn, [training_only])
    assert queue_db.stats(conn)["unscored"] == 12 and "t:train" not in {r["id"] for r in queue_db.to_score(conn, 99)}


def test_queue_claims_newest_first_and_never_twice(conn):
    queue_db.insert(conn, rows(20))
    queue_db.insert(conn, rows(10, start=100))  # received later
    a = queue_db.claim(conn, "a", 10)
    b = queue_db.claim(conn, "b", 100)
    ids_a, ids_b = {r["id"] for r in a["reviews"]}, {r["id"] for r in b["reviews"]}
    assert ids_a == {f"t:{i}" for i in range(100, 110)} and len(ids_b) == 20 and not ids_a & ids_b
    assert a["reviews"][0]["complaints"] == ["food", "ground service"] and a["reviews"][0]["flown"] == "2019-03-01"
    assert queue_db.stats(conn)["claimed"] == 30


def test_queue_done_and_release(conn):
    queue_db.insert(conn, rows(20))
    queue_db.claim(conn, "ok", 5)
    queue_db.claim(conn, "bad", 5)
    assert queue_db.finish(conn, "ok", True) == 5
    assert queue_db.finish(conn, "bad", False) == 5
    assert queue_db.finish(conn, "ok", True) == 0  # already done
    s = queue_db.stats(conn)
    assert (s["pending"], s["claimed"], s["trained"]) == (15, 0, 5)
    assert len(queue_db.claim(conn, "next", 3, replay=10)["replay"]) == 5


def test_queue_releases_stale_claims(conn):
    queue_db.insert(conn, rows(4))
    queue_db.claim(conn, "dead", 4)
    conn.execute("UPDATE queue.reviews SET claimed_at = now() - interval '5 hours'")
    assert queue_db.release_stale(conn) == 4 and queue_db.stats(conn)["pending"] == 4


def test_queue_live_scoring_goes_in_arrival_order_once(conn):
    queue_db.insert(conn, rows(10))
    first = queue_db.to_score(conn, 3)
    assert [r["id"] for r in first] == ["t:0", "t:1", "t:2"]
    for r in first:
        queue_db.mark_scored(conn, r["id"])
    assert [r["id"] for r in queue_db.to_score(conn, 2)] == ["t:3", "t:4"]
    assert queue_db.stats(conn)["unscored"] == 7


def test_queue_schema_is_closed_to_other_users(conn):
    user = f"reader_{uuid.uuid4().hex[:8]}"
    conn.execute(f"CREATE ROLE {user} LOGIN PASSWORD 'x'")
    try:
        conn.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {user}")
        assert conn.execute("SELECT has_schema_privilege(%s, 'queue', 'USAGE')", (user,)).fetchone()[0] is False
    finally:
        conn.execute(f"DROP OWNED BY {user}")
        conn.execute(f"DROP ROLE {user}")

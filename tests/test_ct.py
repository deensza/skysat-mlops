"""Checks for step 8 (continuous training) on fake data.

The 5 standard candidates train, save and predict like the API does. The queue tests need PostgreSQL
(CI starts one; locally `docker compose up -d postgres`) and are skipped without it. They use their
own table, so they never touch real CT jobs.
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("torch")
psycopg = pytest.importorskip("psycopg")

from skysat import schema
from skysat.data import clean
from tests.test_data_and_drift import make_raw
from training import ct_queue
from training.ct_queue import CANDIDATES
from training.train import load_bundle, model_file, predict_proba, run_training


@pytest.fixture(scope="module")
def data():
    df = clean(make_raw(n=900, seed=5))
    df[schema.TARGET] = ((df["online_boarding"] + df["seat_comfort"]) >= 6).astype(int)
    return df.iloc[:700], df.iloc[700:]


def test_every_candidate_trains_and_loads_like_the_api_does(data, tmp_path):
    assert len(CANDIDATES) == 5
    train, val = data
    for name, params in CANDIDATES.items():
        path, metrics = run_training(train, val, {**params, "epochs": 5, "batch_size": 128}, tmp_path / name)
        assert metrics["roc_auc"] > 0.8, name
        bundle = load_bundle(model_file(Path(path).parent))
        assert bundle["meta"]["params"]["model"] == params["model"]
        p = predict_proba(bundle, val)
        assert p.shape == (len(val),) and ((p >= 0) & (p <= 1)).all(), name
        # one passenger at a time (what /predict does) gives the same answer as the batch
        assert predict_proba(bundle, val.iloc[[0]])[0] == pytest.approx(p[0], abs=1e-5), name


@pytest.fixture
def conn(monkeypatch):
    monkeypatch.setattr(ct_queue, "TABLE", "ct_jobs_test")
    try:
        connection = ct_queue.connect()
    except psycopg.OperationalError:
        pytest.skip(f"no PostgreSQL at {os.environ.get('DATABASE_URL', 'localhost:5432')}")
    connection.execute("DROP TABLE IF EXISTS ct_jobs_test")
    ct_queue.ensure_table(connection)
    yield connection
    connection.execute("DROP TABLE ct_jobs_test")
    connection.close()


def test_queue_hands_out_the_oldest_job_once(conn):
    first = ct_queue.add(conn, "rf", {"model": "random_forest"})
    second = ct_queue.add(conn, "gb", {"model": "gradient_boosting"})
    assert ct_queue.claim(conn) == (first, "rf", {"model": "random_forest"})
    assert ct_queue.claim(conn)[0] == second
    assert ct_queue.claim(conn) is None  # both are running now: nobody else gets them


def test_queue_records_results_and_requeues_on_stop(conn):
    job = ct_queue.add(conn, "mlp", {"model": "mlp"})
    ct_queue.claim(conn)
    assert ct_queue.requeue_running(conn, job) == 1  # the worker was stopped mid-job
    assert ct_queue.claim(conn)[0] == job
    ct_queue.finish(conn, job, "done", roc_auc=0.97, version="3", note="registered as version 3")
    (row,) = ct_queue.recent(conn)
    assert row[:5] == (job, "mlp", "done", 0.97, "3")


def test_queue_rejects_unknown_models_and_features(conn):
    with pytest.raises(ValueError, match="Unknown model"):
        ct_queue.add(conn, "x", {"model": "xgboost"})
    with pytest.raises(ValueError, match="Unknown drop_features"):
        ct_queue.add(conn, "x", {"model": "mlp", "drop_features": ["gendr"]})
    assert ct_queue.recent(conn) == []

"""Step 8 – Continuous training (CT) worker: runs in the background and trains every job in the CT queue.

    make start-ct                                  # start it (Docker service ct-worker, restarts on crash)
    make add-ct ARGS="--model random_forest"       # queue a model (training/ct_queue.py); ARGS="--all": 5 models
    make list-ct                                   # jobs and results;  docker compose logs -f ct-worker: live log
    make stop-ct                                   # stop it; a job in progress goes back to the queue

One job at a time, oldest first:
    1. train on data/processed/train.csv, measure on validation.csv (training.train.run_training)
    2. log an MLflow run (tags purpose=ct, ct_job, model_type) with params, metrics and the model file
    3. quality gate from track.py: registered if roc_auc >= 0.95, `champion` only if it beats the current one;
       the API copies switch to a new champion within ~10 s
    4. write the result back to the queue (done / failed + roc_auc + registered version)
A failing job is marked failed and the worker goes on with the next one.
Only one worker is meant to run: at start it puts jobs left `running` by a crash back in the queue.
"""

import os
import signal
import tempfile
import time
import traceback
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from training import ct_queue
from training.track import EXPERIMENT, git_commit, load_data, register
from training.train import DEFAULT_PARAMS, run_training

POLL_SECONDS = float(os.environ.get("CT_POLL_SECONDS", "5"))


class Stop(Exception):
    """Raised on SIGTERM (docker compose stop) or Ctrl+C, so a running job can go back to the queue."""


def _stop(signum, frame):
    raise Stop()


def train_job(client: MlflowClient, job_id: int, name: str, overrides: dict) -> tuple[float, str | None]:
    """Train one job, log it to MLflow, apply the quality gate. Returns (roc_auc, registered version or None)."""
    train_df, val_df = load_data()  # read per job, so a new `make data` is used without a restart
    params = {**DEFAULT_PARAMS, **overrides}
    tags = {"git_commit": git_commit(), "purpose": "ct", "ct_job": str(job_id), "model_type": params["model"]}
    with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(run_name=f"ct-{job_id}-{name}", tags=tags) as run:
        mlflow.log_params(params)
        mlflow.log_params({"train_rows": len(train_df), "validation_rows": len(val_df)})
        path, metrics = run_training(train_df, val_df, params, Path(tmp))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(path, artifact_path="model")
        version = register(client, run, metrics["roc_auc"])
    return metrics["roc_auc"], version


def main():
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050"))
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    with ct_queue.connect() as conn:
        ct_queue.ensure_table(conn)
        if n := ct_queue.requeue_running(conn):
            print(f"{n} job(s) left running by an earlier stop went back to the queue")
        print(f"CT worker ready, checking the queue every {POLL_SECONDS:g} s")
        try:
            while True:
                job = ct_queue.claim(conn)
                if job is None:
                    time.sleep(POLL_SECONDS)
                    continue
                job_id, name, overrides = job
                print(f"\n== job {job_id}: {name} {overrides} ==", flush=True)
                try:
                    auc, version = train_job(client, job_id, name, overrides)
                except Stop:
                    ct_queue.requeue_running(conn, job_id)
                    print(f"job {job_id} back in the queue")
                    raise
                except (Exception, SystemExit) as err:  # load_data() exits when data/processed is missing
                    traceback.print_exc()
                    ct_queue.finish(conn, job_id, "failed", note=str(err)[:500])
                    print(f"job {job_id} failed: {err}")
                    continue
                note = f"registered as version {version}" if version else "rejected by the quality gate"
                ct_queue.finish(conn, job_id, "done", roc_auc=auc, version=version, note=note)
                print(f"job {job_id} done: roc_auc {auc:.4f}, {note}", flush=True)
        except Stop:
            print("CT worker stopped")


if __name__ == "__main__":
    main()

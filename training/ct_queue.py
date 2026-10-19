"""Step 8 – The CT queue: training jobs wait in a PostgreSQL table until the CT worker picks them up.

    python -m training.ct_queue add --model random_forest              # queue one model
    python -m training.ct_queue add --model mlp --rating-one-hot --epochs 30 --name mlp-30-epochs
    python -m training.ct_queue add --all                              # the 5 standard candidates
    python -m training.ct_queue list                                   # latest jobs and their results
(or from the project root: make add-ct ARGS="--model random_forest", make list-ct; both run in Docker)

Anyone can add jobs (you, the CT workflow, a cron job); the worker (training/ct_worker.py) takes the
oldest queued job with SELECT ... FOR UPDATE SKIP LOCKED, so a job is never trained twice.
PostgreSQL already runs for the prediction log (step 6), so the queue needs no extra service.
"""

import argparse
import os

import psycopg
from psycopg.types.json import Jsonb

from skysat import schema
from training.train import SKLEARN_MODELS

TABLE = "ct_jobs"
MODELS = ["mlp", *SKLEARN_MODELS]

# The 5 standard candidates (`add --all`). Linear models cannot learn the step-shaped ratings, so logistic
# regression gets rating_one_hot; the trees split on the raw 0-5 values by themselves.
CANDIDATES = {
    "logistic_regression": {"model": "logistic_regression", "rating_one_hot": True},
    "random_forest": {"model": "random_forest"},
    "gradient_boosting": {"model": "gradient_boosting"},
    "mlp": {"model": "mlp"},
    "mlp_rating_one_hot": {"model": "mlp", "rating_one_hot": True},
}


def connect() -> psycopg.Connection:
    url = os.environ.get("DATABASE_URL", "postgresql://skysat:skysat@localhost:5432/skysat")
    return psycopg.connect(url, autocommit=True)


def ensure_table(conn) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id            bigserial PRIMARY KEY,
            name          text NOT NULL,
            params        jsonb NOT NULL,           -- overrides of training.train.DEFAULT_PARAMS
            status        text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed')),
            created_at    timestamptz NOT NULL DEFAULT now(),
            started_at    timestamptz,
            finished_at   timestamptz,
            roc_auc       double precision,
            model_version text,                     -- registered version, NULL if the quality gate said no
            note          text
        )""")


def add(conn, name: str, params: dict) -> int:
    if params.get("model", "mlp") not in MODELS:
        raise ValueError(f"Unknown model {params['model']!r}; use one of {MODELS}")
    if unknown := set(params.get("drop_features", [])) - set(schema.FEATURES):  # fail now, not in the worker
        raise ValueError(f"Unknown drop_features {sorted(unknown)}; use names from skysat/schema.py")
    return conn.execute(f"INSERT INTO {TABLE} (name, params) VALUES (%s, %s) RETURNING id",
                        (name, Jsonb(params))).fetchone()[0]


def claim(conn) -> tuple[int, str, dict] | None:
    """Mark the oldest queued job as running and return (id, name, params); None if the queue is empty."""
    return conn.execute(f"""
        UPDATE {TABLE} SET status = 'running', started_at = now()
        WHERE id = (SELECT id FROM {TABLE} WHERE status = 'queued' ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1)
        RETURNING id, name, params""").fetchone()


def requeue_running(conn, job_id: int | None = None) -> int:
    """Put a running job (or, with no id, every running job) back in the queue. Returns how many."""
    where = "status = 'running'" + (" AND id = %s" if job_id is not None else "")
    return conn.execute(f"UPDATE {TABLE} SET status = 'queued', started_at = NULL WHERE {where}",
                        (job_id,) if job_id is not None else ()).rowcount


def finish(conn, job_id: int, status: str, roc_auc: float | None = None, version: str | None = None,
           note: str | None = None) -> None:
    conn.execute(f"UPDATE {TABLE} SET status = %s, finished_at = now(), roc_auc = %s, model_version = %s,"
                 f" note = %s WHERE id = %s", (status, roc_auc, version, note, job_id))


def recent(conn, limit: int = 20) -> list[tuple]:
    return conn.execute(f"SELECT id, name, status, roc_auc, model_version, note, created_at FROM {TABLE}"
                        " ORDER BY id DESC LIMIT %s", (limit,)).fetchall()


def main():
    parser = argparse.ArgumentParser(description="Add training jobs to the CT queue, or list them")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("add", help="queue a model for the CT worker")
    a.add_argument("--all", action="store_true", help="queue the 5 standard candidates")
    a.add_argument("--model", choices=MODELS, default="mlp")
    a.add_argument("--name", help="label for the job (default: the model name)")
    a.add_argument("--rating-one-hot", action="store_true", help="add one-hot columns for every 0-5 rating")
    a.add_argument("--drop-features", nargs="*", metavar="NAME", help="schema feature names to leave out")
    a.add_argument("--epochs", type=int, help="MLP only")
    a.add_argument("--lr", type=float, help="MLP only")
    a.add_argument("--seed", type=int)
    ls = sub.add_parser("list", help="latest jobs and their results")
    ls.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    with connect() as conn:
        ensure_table(conn)
        if args.command == "list":
            print(f"{'id':>4}  {'name':<22}{'status':<9}{'roc_auc':>8}  {'version':<8} note")
            for job_id, name, status, auc, version, note, _ in recent(conn, args.limit):
                print(f"{job_id:>4}  {name:<22}{status:<9}{'' if auc is None else f'{auc:.4f}':>8}  "
                      f"{version or '':<8} {note or ''}")
            return
        if args.all:
            jobs = CANDIDATES
        else:
            params = {"model": args.model, "rating_one_hot": args.rating_one_hot}
            optional = {"drop_features": args.drop_features, "epochs": args.epochs, "lr": args.lr, "seed": args.seed}
            params.update({k: v for k, v in optional.items() if v is not None})
            jobs = {args.name or args.model + ("_rating_one_hot" if args.rating_one_hot else ""): params}
        for name, params in jobs.items():
            try:
                print(f"queued job {add(conn, name, params)}: {name} {params}")
            except ValueError as err:
                raise SystemExit(f"ERROR: {err}")


if __name__ == "__main__":
    main()

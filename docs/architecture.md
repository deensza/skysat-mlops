# SkySat architecture

The map of the project: the pieces, how they talk, which command starts what. Model details: [llm.md](llm.md).

## 1. What SkySat does

An airline gets thousands of free-text reviews. SkySat reads one and answers: score out of 10, recommended yes/no, main
complaint. The model is one LoRA adapter on a frozen 1.5B LLM. Passengers' reviews (simulated from the 128k dataset)
arrive at the server's **intake** and wait in a **queue**. A **trainer** on a rented **RunPod GPU** takes batches from
the queue, trains, gates, and registers in **MLflow**. The API serves the **champion**, logs every answer to
**PostgreSQL**, **Grafana** shows it, **drift** is checked on the live reviews' true labels. **CI/CD** tests and
deploys; a drift alarm can start the training by itself (built, not yet switched on).

## 2. The big picture

```mermaid
flowchart LR
    sim["simulator (a PC)<br/>simulator/replay.py"]
    subgraph SRV["Server (docker compose)"]
        nginx["Nginx<br/>/score /model /health /docs /reviews<br/>/queue/* (login trainer)"]
        intake["intake<br/>llm/intake.py"]
        queue[("PostgreSQL<br/>queue.reviews")]
        api["api<br/>llm/serve.py, CPU"]
        pg[("PostgreSQL<br/>llm_scores")]
        grafana["Grafana"]
    end
    pod["RunPod pod<br/>llm/trainer.py<br/>deletes itself"]
    mlflow[("MLflow (DagsHub)<br/>skysat-reviews@champion")]
    subgraph GH["GitHub (no GPU)"]
        ci["ci.yml<br/>tests, image, deploy"]
        drift["drift.yml<br/>drift check, issue"]
        train["train.yml<br/>cooldown, queue check, rents the pod"]
    end
    sim -- "POST /reviews" --> nginx --> intake --> queue
    intake -- "each review + true labels" --> api
    pod -- "claim / done" --> nginx
    pod -- "adapter, if it beats the champion" --> mlflow
    mlflow -- "adapter, checked every minute" --> api
    api -- "numbers only" --> pg --> grafana
    drift -- "GET /scores/recent, GET /model" --> nginx
    drift -- "DRIFT and AUTO_RETRAIN=true" --> train
    train -- "llm.runpod_pod --wait" --> pod
    ci -- "image, rolls back if the smoke test fails" --> SRV
```

Ideas that hold it together:

- **The adapter is not inside the image.** The API asks MLflow which version is `champion` and loads it next to the base
  model it already has. A new model goes live without a redeploy.
- **The queue decouples arriving data from training.** Reviews keep arriving while nothing trains; a batch that fails
  goes back to pending.
- **The GPU is rented only while it trains.** `train.yml` (started by `drift.yml` or by hand), `make train-remote` or
  the intake with `AUTO_TRAIN=1` starts a pod on the cheapest free GPU; the pod deletes itself when the trainer exits,
  and a timer inside the pod deletes it if it does not. With `--wait` (the workflow and `make train-remote`) the
  launcher also deletes the pod on any way out and `--sweep` removes strays; a pod the intake starts has only the first
  two.
- **Review text stays private.** It is in the queue's own schema, which Grafana's user cannot read on the server (the
  local stack's Grafana uses the database owner); monitoring stores numbers only.

## MLOps level

Google's "MLOps: Continuous delivery and automation pipelines in machine learning" names three levels.

| Level | What it asks for | What SkySat has |
|---|---|---|
| 0, manual | Every step by hand: data, training, hand-over to the people who deploy | Not here: the steps are scripts and workflows |
| 1, ML pipeline automation | Training is a pipeline that runs by itself on new data (continuous training), with a gate and a registry | Queue (`llm/queue_db.py`) -> `llm/trainer.py` -> `llm/gate.py` -> MLflow registry, started by `train.yml` |
| 2, CI/CD pipeline automation | The pipeline's own code is built, tested and deployed automatically, and the loop runs without a person | `ci.yml` (tests, image, deploy with rollback) and `drift.yml` -> `train.yml` |

What makes it level 2 here:

- A code change goes through tests (`tests/`, `nginx -t`), an image on ghcr.io, and a deploy on the server; if the smoke
  test fails the deploy job goes back to the image that was running.
- Drift starts the training: `drift.yml` checks the serving model, opens an issue and, if `AUTO_RETRAIN` is `true`,
  calls `train.yml`, which rents a RunPod pod, trains, gates, and registers in MLflow. The API loads the new champion
  by itself.
- The pipeline itself is tested in CI: `tests/test_pipeline.py` runs the real trainer against a real PostgreSQL with
  only the GPU training replaced by a stub, and checks the queue's state afterwards.
- Cost is bounded: a cooldown, a check that at least 300 reviews are pending, a 45 minute limit per pod, and a sweep
  that fails the run if a pod is left.

What a real level 2 system has that this one does not:

- No orchestrator such as Kubeflow or Airflow: GitHub Actions plays that part.
- No feature store, and data validation is only the intake's request checks.
- One small server and no staging environment: the new image goes straight to the only server.
- The loop is built but must be switched on by the repository owner ([server.md](server.md)); it has not run on GitHub
  yet.

## 3. The services (docker-compose.yml)

| Service | What | Port |
|---|---|---|
| `api` | `llm/serve.py`: base model + adapter, one review at a time, `mem_limit` 8 GB, base model cached in the `llm-models` volume | internal 8000 |
| `intake` | `llm/intake.py` (same image): `POST /reviews` -> queue, each live review to `api` with its true labels, `/queue/*` for the trainer | 127.0.0.1:8001 |
| `nginx` | gateway; routes `/score`, `/model`, `/health`, `/docs`, `/scores/recent`, `/reviews`; on the server also `/queue/*` for the `trainer` login | 8080 |
| `mlflow` | local registry (or the shared one from `.env`) | 5050 |
| `postgres` | `llm_scores` (`monitoring/init.sql`) and the schema `queue` | 5432 |
| `grafana` | dashboard `skysat`, anonymous read-only | 3000 |
| `k6` | `make d-loadtest`, 1 user for 1 minute against `/score` | on demand |

On the server, [docker-compose.server.yml](../docker-compose.server.yml) adds Caddy (HTTPS), the login, the rate limit,
the read-only Grafana database user and the restart policies ([server.md](server.md)).

## 4. Life of a model

1. `make simulate-seed N=2000`: 2,000 reviews flown <= 2018 go into the queue (not sent as live traffic).
2. `make train-remote` (RunPod, RTX 3070 at $0.13/h) or `make llm-train` (this PC): the trainer claims the newest pending
   reviews, keeps 10% back for the gate, and LoRA-tunes on the rest (1,800 reviews, 322 s on an RTX 4070 Laptop).
3. The gate (`llm/gate.py`): MAE <= 1.0, recommended >= 0.90, complaint >= 0.65. v1 from the queue: 0.88, 92.0%, 71.5%.
4. Logged to MLflow (parameters, metrics, the 17 MB adapter), registered as `skysat-reviews`; it becomes `champion` if
   no champion exists or it beats the champion on the same gate reviews (`combined`). The batch is marked trained if a
   version was registered; if the gate fails it goes back to pending.
5. The API sees the new alias within `MODEL_RELOAD_SECONDS`, loads the adapter, deletes the old one.
6. Drift: `make simulate ... SCENARIO=harsh` sends reviews a harsher survey scored; the live MAE rises past the gate,
   and the next batch (those newest reviews) trains a v2 that beats v1 on them.

## 5. Life of a request

`POST /reviews` → Nginx → `intake` → one row in `queue.reviews` (pending). The intake's loop sends it to `POST /score`
with its true labels → `api` (queues behind the lock, generates greedily, parses three lines) → answer (`score`,
`recommended`, `complaint`, `model_version`, `latency_ms`) → one row in `llm_scores` (numbers and fixed labels).
Malformed answers are `null` fields and count as wrong in the live measures.

## 6. CI, CD and the schedule

All three workflows are code and configuration only: none has run on GitHub yet.

- `ci.yml`: pytest (no torch; the queue's SQL and `tests/test_pipeline.py` against a PostgreSQL service), `nginx -t` on
  both gateway configs, `docker compose config` for both compose files, build the `llm/Dockerfile` image. On `main` it
  is pushed to ghcr.io and, if the server's runner and `DEPLOY_TO_RUNNER` are set, deployed and smoke-tested. The
  deploy job remembers the running image and rolls back to it if the smoke test fails (only the image, not the compose
  or gateway files). It does not set the `MLFLOW_*` values itself. It never runs for pull requests.
- `drift.yml`: scheduled hourly at :35. `llm/drift_check.py` asks `/model` which version is serving and judges only that
  version's answers in `/scores/recent?version=N` (`--all-versions` counts all): after a promotion the old model's
  answers no longer count, so a problem that is already fixed does not ask for another retraining. On DRIFT it opens one
  issue "Drift detected: the serving model fell below the gate" and, only if the repository variable `AUTO_RETRAIN` is
  `true`, calls `train.yml` with `--replay 0 --epochs 3`: the new reviews only, because after drift the old labels are
  the problem (with the default, 500 replayed reviews, the retraining of 16 Nov failed the gate; without them it
  passed, see [llm.md](llm.md#the-gate-and-what-it-measured)). A scheduled run with missing secrets is skipped with a
  warning.
- `train.yml` ("Train"): started by `drift.yml` or by hand. It checks its settings, respects a cooldown (variable
  `TRAIN_COOLDOWN_HOURS`, default 6: no automatic run if a GPU was rented less than that long ago, by either workflow,
  whether that run worked or not), reads the
  queue and rents nothing if fewer than 300 reviews are pending, runs
  `python3 -m llm.runpod_pod --wait --branch main --max-minutes 45`, then `--sweep`, which fails the run if a trainer
  pod is still there, and writes a summary table (queue and serving model before and after, registered, version,
  promoted). One training run at a time (`concurrency: train`).
- `llm/runpod_pod.py --wait` returns the trainer's exit code and writes `exit_code`, `registered`, `version`, `promoted`
  for the workflow. The launcher deletes the pod on any way out (deadline, Ctrl-C, a cancelled workflow, a network
  error) with three tries, and a network blip does not end the wait. The pod waits 45 s after the trainer ends so the
  launcher can read the last log lines.
- The settings the loop needs (secrets and variables) are listed in [server.md](server.md#switching-it-on).

## 7. Where things live

| What | Where |
|---|---|
| Review dataset | this PC (`data/reviews/`, gitignored); on the server only in the queue |
| Base model | Hugging Face, cached in the `llm-models` Docker volume (server) and on the pod's disk |
| Adapters, runs, metrics | MLflow (`artifacts/llm-runs/mlflow.db` locally, DagsHub when shared) |
| Prediction log | PostgreSQL `llm_scores` |
| Secrets | `.env` (local), `/opt/skysat/.env` and `gateway-users` (server), GitHub secrets, RunPod pod env; never in git |

## 8. Rules that shape the design

- No review text or prompt in git, images, MLflow or the monitoring table.
- Every number in the docs was measured (see [llm.md](llm.md)).
- A new adapter must beat the champion on the same reviews; otherwise the champion stays.
- A pod must never outlive its job: self-delete, an in-pod timer, and the launcher's deadline and sweep
  (`--wait` only: a pod started by the intake has the first two).

## 9. Known limits

- CPU serving takes seconds per review (15-24 s on a laptop) and handles one at a time, so live scoring lags behind a
  fast simulator; the API needs ~8 GB of RAM (the 1 GB server VM of the first version cannot host it).
- Real 2019-2022 drift does not hurt the model, so the drift demo uses the simulated `harsh` scenario.
- The pod clones the code from GitHub: a branch must be pushed before it can train on it.
- A review claimed for training before the intake has scored it is not scored any more.
- The automatic loop (`drift.yml` -> `train.yml`) needs the secrets and variables in
  [server.md](server.md#switching-it-on)
  and `AUTO_RETRAIN=true`; it is built but not yet switched on, and it has not run on GitHub yet.
- A full training run on RunPod has not been done yet: only a boot test pod has (it loaded the base model and deleted
  itself). The first trainings from the server's queue ran on a Colab T4.
- The deploy rollback restores only the image, not the compose or gateway files.
- There is one server and no staging environment.

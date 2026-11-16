# SkySat – MLOps Term Project

SE4458 Software Architecture & Design of Modern Large Scale Systems – Assignment 1 (Demo)
Topic: **MLOps – Build a simple ML model and demonstrate its complete lifecycle from training to production**

Team: Bermal Deniz Akkaya, Osman Şahin Güler

SkySat is a **complaint box** for airline reviews. A small LLM (Qwen2.5-1.5B-Instruct, LoRA-tuned) reads a real
review and answers with the score out of 10, whether the reviewer would recommend the airline, and the main complaint.
Passengers' reviews reach the server, wait in a queue, and a rented GPU retrains the model from it:

```
passengers (simulated) ──► POST /reviews ──► queue (PostgreSQL) ──► trainer (RunPod GPU, $0.13/h) ──► MLflow champion ──► API
                              └─ each review, with its true labels ──► API ──► Grafana, drift check
```

Around that one model the project shows the full MLOps lifecycle:
data intake → queue → LoRA training on a rented GPU → quality gate → experiment tracking & model registry (MLflow) →
packaging (Docker) → serving (FastAPI, Nginx) → monitoring (PostgreSQL, Grafana) → drift detection → retraining →
CI/CD (GitHub Actions). Details: [docs/llm.md](docs/llm.md).

Measured on the server on 16 Nov 2026, with the champion (version 1) serving:

| | 100 healthy live reviews | 300 "harsh" live reviews | Retrained on that batch: 100 held-back reviews (champion on the same) |
|---|---|---|---|
| Score MAE | 0.95 | 1.15 (drift check: DRIFT) | 1.07 (1.18) |
| Recommended accuracy | 95.0% | 89.0% | 87.0% (85.0%) |

The retrained adapter was better than the champion on the drifted reviews but failed the gate (MAE <= 1.0, recommended
>= 90%), so it was not registered and version 1 still serves: the drift was detected, not yet fixed. Why, and the
earlier runs: [docs/llm.md](docs/llm.md#the-gate-and-what-it-measured).

## Data and licence

- **Data:** [Airline Reviews (128k)](https://www.kaggle.com/datasets/joelljungstrom/128k-airline-reviews) by joelljungstrom
  on Kaggle, licence DbCL-1.0, on this PC in `data/reviews/` ([data/README.md](data/README.md)). The simulator posts its
  reviews to the server as passengers would.
- The dataset is **never committed, copied into Docker images, logged to MLflow, or written to the table Grafana reads**.
  On the server it lives only in the private queue. The tests use made-up reviews.
- Base model: Qwen2.5-1.5B-Instruct (Apache-2.0), downloaded at the first start, never committed.
- Code: [MIT License](LICENSE).

## Project structure

| Folder | Purpose |
|---|---|
| `llm/` | Intake (`intake.py`: `POST /reviews` + the queue API), queue (`queue_db.py`), trainer (`trainer.py`), RunPod launcher (`runpod_pod.py`), training (`train_lora.py`), gate (`gate.py`), data (`reviews.py`), API (`serve.py`), drift check, Dockerfile |
| `simulator/` | `replay.py`: simulated passengers post reviews to `POST /reviews`; `scenarios.py`: simulated drift |
| `monitoring/` | Grafana dashboard, database setup (`llm_scores`), watchdog |
| `nginx/` | Gateway configuration (local and server) |
| `loadtest/` | k6 test for `POST /score` |
| `tests/` | Automated tests (pytest, no GPU; the queue's SQL against a real PostgreSQL in CI) |
| `docs/` | [LLM](docs/llm.md), [architecture](docs/architecture.md), [server](docs/server.md), [presentation](docs/presentation.md) |
| `demo/` | [Demo runbook](demo/README.md) |
| `.github/workflows/` | CI/CD, drift check |

## Setup

You need Git, Docker, `make` and [uv](https://docs.astral.sh/uv/). Training needs an NVIDIA GPU: a RunPod pod
(`RUNPOD_API_KEY`) or a local card.

```bash
make install        # .venv: pinned packages for the tests and the simulator (no torch)
make test           # should end with "passed"
make llm-install    # .venv-llm: CUDA PyTorch, transformers, peft (local GPU only, ~5 GB)
kaggle datasets download joelljungstrom/128k-airline-reviews -p data/reviews --unzip   # once
```

`make help` lists every command.

## Run the lifecycle

```bash
make llm-up                                   # API, intake, MLflow, PostgreSQL, Nginx, Grafana
make simulate-seed N=2000                     # the first model's training reviews, into the queue
make llm-train                                # v1 on this PC's GPU (or: make train-remote, on RunPod)
make simulate FROM=2019-01 PER_MONTH=100      # live passengers -> Grafana (http://localhost:3000)
make simulate FROM=2019-02 PER_MONTH=300 SCENARIO=harsh   # the drift
make llm-drift                                # drift check on the last 24 h of the log
make llm-train                                # v2 from the newest reviews; promoted if it beats v1
```

`make llm-train` uses a local sqlite MLflow unless `SHARED=yes`. Docker: `make d-ps`, `d-logs`, `d-build`,
`d-down` (data kept), `d-wipe` (deletes local data, asks first).

## MLOps level

By Google's three levels (manual, ML pipeline automation, CI/CD pipeline automation), SkySat is built to level 2: tests,
image and deploy with rollback, and a drift alarm that starts the training workflow. The loop is built but not yet
switched on (the repository owner adds the secrets and variables) and has not run on GitHub yet. What a real level 2
system has that this one does not: [docs/architecture.md](docs/architecture.md#mlops-level).

## Pipeline

| Stage | Where |
|---|---|
| Intake | `intake` service (`llm/intake.py`): `POST /reviews` into `queue.reviews`; each live review to `/score` with its true labels |
| Train + gate + register | `llm/trainer.py` on a RunPod GPU (`llm/runpod_pod.py`) or this PC; MLflow experiment `skysat-llm`, model `skysat-reviews`, alias `champion` |
| Serve | `api` service (`llm/serve.py`, CPU, 8 GB), behind Nginx: `/score`, `/model`, `/health`, `/docs` |
| Monitor | `llm_scores` table → Grafana; `GET /scores/recent?hours=N` returns aggregates only |
| Drift | `llm/drift_check.py`: DRIFT if the live MAE of the serving model > 1.0 or its recommended accuracy < 0.90 by more than the sampling noise, on >= 100 labelled requests |
| CI/CD | `ci.yml`: tests (including `tests/test_pipeline.py`), `nginx -t`, compose check, image build, deploy on main only, if switched on, with a rollback to the running image if the smoke test fails |
| Retrain | `train.yml`: started by `drift.yml` (only if `AUTO_RETRAIN` is `true`) or by hand; cooldown, needs >= 300 pending reviews, rents a RunPod GPU for at most 45 minutes, then checks no pod is left. Built, not yet switched on, not yet run on GitHub. Also `make train-remote` |

See [docs/architecture.md](docs/architecture.md) for the picture and [docs/server.md](docs/server.md) for the always-on server.

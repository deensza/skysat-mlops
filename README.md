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

Measured on the server on 16 and 17 Nov 2026, live traffic scored by the model that was serving:

| | 100 healthy reviews (version 1) | 300 "harsh" reviews (version 1) | 165 new "harsh" reviews after retraining (version 4) |
|---|---|---|---|
| Score MAE | 0.95 | 1.15 (drift check: DRIFT) | 0.55 (drift check: no drift) |
| Recommended accuracy | 95.0% | 89.0% | 97.0% |

The drift check fired on the harsh reviews. A first retraining that mixed 500 old reviews into the new ones failed the
gate (MAE <= 1.0, recommended >= 90%) and was not registered. A second one, on the new reviews only, passed, beat the
champion on the same held-back reviews, became version 4, and the server loaded it by itself. These trainings of 16
and 17 Nov ran on a Google Colab T4 started by hand, before the RunPod path had been tried. Colab is not a supported
way to train and there is no Colab code in this repository.

On 17 Nov 2026 the Train workflow ran on RunPod: an RTX 3070 at $0.13/h for about 5 minutes (about one cent), started
by hand. It trained on a batch of 300 reviews from the server's queue and registered version 5 (MAE 0.33, recommended
93.0% on 100 held-back reviews). Version 5 did not beat version 4 on those reviews (combined 0.858 against 0.861), so
version 4 stayed the champion.

The same day the automatic chain ran end to end once. The drift check found drift on version 4 (MAE 1.23, recommended
81.2% on 591 requests) and called the Train workflow; a pod on an RTX 3080 Ti ($0.18/h) existed for 5 min 32 s (about
2 cents); version 6 was registered and promoted, and the server loaded it. Three calls before that, between 13:12 and
14:24 UTC, rented nothing: no GPU type in the list was free. Every drift check that day was started by hand, because
GitHub's hourly schedule did not start one.
Not known yet: how version 6 does on live traffic (it had not been judged when this was written).

Why the first retraining failed, and every run: [docs/llm.md](docs/llm.md#the-gate-and-what-it-measured).

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
| `.github/workflows/` | CI/CD (`ci.yml`), drift check (`drift.yml`), training on a rented GPU (`train.yml`) |

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
make llm-train LLM_MLFLOW=http://localhost:5050   # v1 on this PC's GPU, into the stack's MLflow: the API loads it
make simulate FROM=2019-01 PER_MONTH=100      # live passengers -> Grafana (http://localhost:3000)
make simulate FROM=2019-02 PER_MONTH=300 SCENARIO=harsh   # the drift
make llm-drift                                # drift check on the last 24 h of the log
make queue-stats                              # wait until "ready" is 300 or more (scored reviews; ~20 s each on a laptop)
make llm-train LLM_MLFLOW=http://localhost:5050 ARGS="--replay 0 --epochs 3"   # v2 from the new reviews only; promoted if it beats v1
```

Plain `make llm-train` writes to a local sqlite MLflow (`artifacts/llm-runs/mlflow.db`), which the API in Docker does
not read: `LLM_MLFLOW=http://localhost:5050` trains into the stack's own MLflow, `SHARED=yes` into the team's (from
`.env`). After drift, retrain with `--replay 0 --epochs 3`: with the default (500 replayed reviews, one epoch) the
retraining of 16 Nov failed the gate. On RunPod the same trainer is started by the Train workflow or by
`make train-remote`, against the server's queue (the pod needs a public queue address). Docker: `make d-ps`, `d-logs`,
`d-build`, `d-down` (data kept), `d-wipe` (deletes local data, asks first).

## MLOps level

By Google's three levels (manual, ML pipeline automation, CI/CD pipeline automation), SkySat is built to level 2: tests,
image and deploy with rollback, and a drift alarm that starts the training workflow. The loop has been switched on
since 17 Nov 2026 (secrets and variables set, `AUTO_RETRAIN=true`).
That day the chain ran end to end once: a drift alarm called the Train workflow, which rented a GPU, trained, and
promoted version 6. Each drift check was started by hand: GitHub's hourly schedule has not started one yet.
The deploy job is still off (no self-hosted runner, `DEPLOY_TO_RUNNER` unset): the server is updated by hand with
`az vm run-command`. What a real level 2 system has that this one does not:
[docs/architecture.md](docs/architecture.md#mlops-level).

## Pipeline

| Stage | Where |
|---|---|
| Intake | `intake` service (`llm/intake.py`): `POST /reviews` into `queue.reviews`; each live review to `/score` with its true labels |
| Train + gate + register | `llm/trainer.py` on a RunPod GPU (`llm/runpod_pod.py`) or this PC; MLflow experiment `skysat-llm`, model `skysat-reviews`, alias `champion` |
| Serve | `api` service (`llm/serve.py`, CPU, 8 GB), behind Nginx: `/score`, `/model`, `/health`, `/docs` |
| Monitor | `llm_scores` table → Grafana; `GET /scores/recent?hours=N` returns aggregates only |
| Drift | `llm/drift_check.py`: DRIFT if the live MAE of the serving model > 1.0 or its recommended accuracy < 0.90 by more than the sampling noise, on >= 100 labelled requests |
| CI/CD | `ci.yml`: tests (including `tests/test_pipeline.py`), `nginx -t`, compose check, image build; on pushes to main the image goes to ghcr.io. The deploy job (rollback to the running image if the smoke test fails) is written but off: no self-hosted runner, `DEPLOY_TO_RUNNER` unset, so it has never run and the server is updated by hand |
| Retrain | `train.yml`: started by `drift.yml` (only if `AUTO_RETRAIN` is `true`) or by hand; cooldown, needs >= 300 `ready` reviews (pending and already scored by the model as live traffic), rents a RunPod GPU for at most 45 minutes, then checks no pod is left. Switched on since 17 Nov 2026; that day it trained on RunPod by hand and once started by the drift check ([docs/llm.md](docs/llm.md#what-ran-on-runpod-17-nov-2026)). Also `make train-remote` |

See [docs/architecture.md](docs/architecture.md) for the picture and [docs/server.md](docs/server.md) for the always-on server.

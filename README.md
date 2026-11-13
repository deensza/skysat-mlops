# SkySat – MLOps Term Project

SE4458 Software Architecture & Design of Modern Large Scale Systems – Assignment 1 (Demo)
Topic: **MLOps – Build a simple ML model and demonstrate its complete lifecycle from training to production**

Team: Bermal Deniz Akkaya, Osman Şahin Güler

SkySat is a **complaint box** for airline reviews. A small LLM (Qwen2.5-1.5B-Instruct, LoRA-tuned) reads a real review
and answers with the score out of 10, whether the reviewer would recommend the airline, and the main complaint.
Around that one model the project shows the full MLOps lifecycle:
data → LoRA training on a GPU → quality gate → experiment tracking & model registry (MLflow) → packaging (Docker) →
serving (FastAPI, Nginx) → monitoring (PostgreSQL, Grafana) → drift detection → retraining → CI/CD (GitHub Actions).

| | v1 (champion) | under the "harsh" survey, 2019 | v2 (retrained on harsh 2019) |
|---|---|---|---|
| Score MAE | 0.91 | 1.18 (fails the gate) | 0.41 |
| Recommended accuracy | 94.5% | 89.5% | 95.5% |

v1: LoRA r=16 on 2,000 reviews flown <= 2018, 5 min on an RTX 4070, adapter 17 MB; complaint accuracy 68%.
Real 2019-2022 reviews do not hurt it (MAE 0.75-0.83), so the drift in the demo is simulated; v2 is promoted
automatically. Details: [docs/llm.md](docs/llm.md).

## Data and licence

- **Data:** [Airline Reviews (128k)](https://www.kaggle.com/datasets/joelljungstrom/128k-airline-reviews) by joelljungstrom
  on Kaggle, licence DbCL-1.0. The reviews were scraped from airlinequality.com by the dataset's author.
- The dataset is **never committed, copied into Docker images, logged to MLflow or stored as text in PostgreSQL**.
  Download it yourself into `data/reviews/` ([data/README.md](data/README.md)); the tests use made-up reviews.
- Base model: Qwen2.5-1.5B-Instruct (Apache-2.0), downloaded at the first start, never committed.
- Code: [MIT License](LICENSE).

## Project structure

| Folder | Purpose |
|---|---|
| `llm/` | Training (`train_lora.py`), quality gate (`gate.py`), data (`reviews.py`), API (`serve.py`), drift check (`drift_check.py`), Dockerfile |
| `simulator/` | `replay.py` posts live reviews with their true labels; `scenarios.py` simulates drift |
| `notebooks/` | Colab notebook for training on a free T4 GPU |
| `monitoring/` | Grafana dashboard, database setup (`llm_scores`), watchdog |
| `nginx/` | Gateway configuration (local and server) |
| `loadtest/` | k6 test for `POST /score` |
| `tests/` | Automated tests (pytest, no GPU) |
| `docs/` | [LLM](docs/llm.md), [architecture](docs/architecture.md), [server](docs/server.md), [presentation](docs/presentation.md) |
| `demo/` | [Demo runbook](demo/README.md) |
| `.github/workflows/` | CI/CD, traffic replay, drift check |

## Setup

You need Git, Docker, `make` and [uv](https://docs.astral.sh/uv/). Training needs an NVIDIA GPU (or Colab).

```bash
make install        # .venv: pinned packages for the tests and the replay (no torch)
make test           # should end with "passed"
make llm-install    # .venv-llm: CUDA PyTorch, transformers, peft (GPU machine only, ~5 GB)
kaggle datasets download joelljungstrom/128k-airline-reviews -p data/reviews --unzip
```

`make help` lists every command.

## Run the lifecycle

```bash
make llm-cycle      # v1 on <= 2018 reviews, v1 on harsh 2019 (fails the gate), v2 retrained on harsh 2019 (promoted)
make d-up           # the Docker stack: API, MLflow, PostgreSQL, Nginx, Grafana
make smoke          # /health and one POST /score through Nginx (http://localhost:8080)
make llm-replay SCENARIO=harsh MONTHS=1 PER_MONTH=100   # live reviews with true labels -> Grafana (http://localhost:3000)
make llm-drift      # drift check on the last 24 h of the log
```

The `llm-*` training commands use a local sqlite MLflow unless `SHARED=yes`. Docker: `make d-ps`, `d-logs`, `d-build`,
`d-down` (data kept), `d-wipe` (deletes local data, asks first).

## Pipeline

| Stage | Where |
|---|---|
| Train + gate + register | `llm/train_lora.py` on a GPU; MLflow experiment `skysat-llm`, model `skysat-reviews`, alias `champion` |
| Serve | `api` service (`llm/serve.py`, CPU, 8 GB), behind Nginx: `/score`, `/model`, `/health`, `/docs` |
| Monitor | `llm_scores` table → Grafana; `GET /scores/recent?hours=N` returns aggregates only |
| Drift | `llm/drift_check.py`: DRIFT if live MAE > 1.0 or recommended accuracy < 0.90 on >= 50 labelled requests |
| CI/CD | `ci.yml`: tests, compose check, image build, deploy (main only, if switched on) |
| Schedule | `traffic.yml` hourly replay and `drift.yml` hourly drift check (both commented out until the final run); drift opens a GitHub issue |
| Retrain | by a person on a GPU: `make llm-cycle`. GitHub has no GPU |

See [docs/architecture.md](docs/architecture.md) for the picture and [docs/server.md](docs/server.md) for the always-on server.

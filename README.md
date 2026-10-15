# SkySat – MLOps Term Project

SE4458 Software Architecture & Design of Modern Large Scale Systems – Assignment 1 (Demo)
Topic: **MLOps – Build a simple ML model and demonstrate its complete lifecycle from training to production**

Team: Bermal Deniz Akkaya, Osman Şahin Güler

SkySat predicts whether an airline passenger is satisfied and shows the full MLOps lifecycle around that model:
data → training → experiment tracking & model registry (MLflow) → packaging (Docker) → serving (FastAPI) →
scaling & load testing (Nginx, k6) → monitoring (PostgreSQL, Grafana) → drift detection → retraining → CI/CD (GitHub Actions).

Dataset: [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
(the original data behind Kaggle Playground Series S6E10, *Predicting Airline Satisfaction*).

## Data and licence

- **Competition data:** [Playground Series – Season 6, Episode 10: Predicting Airline Satisfaction](https://www.kaggle.com/competitions/playground-series-s6e10),
  sponsored by Google LLC and hosted by Kaggle. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Original dataset:** linked above; see its Kaggle page for its own licence.
- The competition rules forbid redistributing the data, so it is **never committed, copied into Docker images, or used in CI**.
  Download it yourself into `data/` (see [data/README.md](data/README.md)); tests use small synthetic data instead.
- Code for this competition stays inside the team; this repository is kept private while the competition is running.

## Project structure

| Folder | Purpose |
|---|---|
| `data/` | Dataset files (downloaded, not committed) |
| `simulator/` | Sends passenger records to the API (normal and drift modes) |
| `training/` | Training and retraining scripts, MLflow logging |
| `api/` | FastAPI prediction service |
| `monitoring/` | Grafana dashboards and database setup |
| `loadtest/` | k6 load test scripts |
| `nginx/` | Load balancer configuration |
| `tests/` | Automated tests (pytest) |
| `.github/workflows/` | CI/CD pipeline (GitHub Actions) |

## Training on Kaggle (competition track)

A separate, optional track for the Kaggle leaderboard. It uses the competition files, not the project pipeline below.
The model is a PyTorch MLP ([training/train_torch.py](training/train_torch.py)).
One-time: `pip install kaggle`, add your Kaggle API credentials, join the competition and accept its rules.

```bash
export KAGGLE_USERNAME=<your kaggle username>
make kaggle-train   # pushes a private notebook, waits for it, downloads artifacts/kaggle/model.pt + submission.csv
```

The competition data stays on Kaggle; the notebook is private.

## Setup

Requirements: Git, Docker Desktop, Python 3.12.

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```powershell
# Windows (PowerShell)
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then download the dataset as described in [data/README.md](data/README.md).

## Step 2 – Data preparation and traffic simulator

```bash
# clean the data and split it: processed/train.csv, validation.csv, live.csv
python -m training.prepare_data

# see how the "winter storm" drift changes the passengers
python -m simulator.simulate compare --n 5000

# write passengers to a CSV (normal or drift mode)
python -m simulator.simulate generate --mode normal --n 2000
python -m simulator.simulate generate --mode drift  --n 2000

# later (step 4+): send passengers to the API instead of a file
python -m simulator.simulate generate --mode drift --n 500 --url http://localhost:8080/predict
```

- `data/train.csv` is split 80/20 (stratified) into **train** and **validation**.
- `data/test.csv` becomes **live** traffic: the model never sees these passengers during training.
- **Drift mode – winter storm:** 60% of passengers get an extra 30–180 min delay, and 7 ratings that depend on
  the airport, the schedule and the online systems drop by 2 stars (never below 1; 0 = not rated stays 0).
  The labels do not change, so a model trained before the storm becomes less accurate until it is retrained.

## Step 3a – Training

```bash
make install         # once: .venv with the pinned packages (Python 3.12)
make install-train   # once: adds PyTorch
make data            # step 2: clean + split (needs data/train.csv and test.csv)
make train           # trains on data/processed/train.csv, evaluates on validation.csv
```

Writes `artifacts/model.pt` (gitignored) and prints validation AUC, precision, recall, F1 and the majority-class baseline.
`model.pt` is one file: network weights plus the preprocessing settings. The API will load it with `load_bundle()` and
`predict_proba()` from [training/train.py](training/train.py), the same code that trained it.
Step 3b (MLflow) calls `run_training()` and logs / registers the result. `make help` lists all shortcuts; `make ci` runs the CI checks locally.

## Step 3b – MLflow tracking and model registry

```bash
docker compose up -d mlflow     # UI: http://localhost:5050 (5000 is taken by AirPlay on macOS)
python -m training.track        # train + log to MLflow + register (needs make data and make install-train)
python -m training.track --stub --stub-auc 0.97   # fake training: tries the registry without data or PyTorch
```

- Each run logs the parameters, validation metrics, tags (`git_commit`, `purpose=training`) and `model/model.pt`.
- **Quality gate:** a model is registered as `skysat-satisfaction` only if validation **ROC AUC ≥ 0.95**
  (the LightGBM baseline reaches ~0.958).
- The `champion` alias moves to the new version only if its ROC AUC is higher than the current champion's.
  The API (step 4) loads `models:/skysat-satisfaction@champion`, so a better model goes live without a code change.
- Runs and models are kept in the `mlflow-data` Docker volume. `docker compose down -v` deletes them.

## Step 4 – Prediction API

```bash
docker compose up -d --build       # MLflow + API (+ Nginx from step 5); Swagger: http://localhost:8080/docs
curl localhost:8080/health         # {"status":"ok","model_version":"1",...}
curl localhost:8080/model          # serving version + its validation metrics
```

- FastAPI service in [api/main.py](api/main.py): `POST /predict`, `GET /health`, `GET /model`.
- The request body is generated from [skysat/schema.py](skysat/schema.py): unknown categories, ratings outside 0–5,
  missing fields or an extra `satisfied` label are rejected with 422.
- The model is **not** inside the image. At start the API downloads `skysat-satisfaction@champion` from MLflow and
  checks the alias every 10 s; a new champion is swapped in while requests keep being served.
  If MLflow is down, the API keeps serving the model it already has.
- Preprocessing comes from the same `training/train.py` that trained the model (no training/serving skew).

## Step 5 – Scaling and load test

```bash
docker compose up -d --build                     # MLflow + 3 API copies + Nginx on http://localhost:8080
docker compose --profile loadtest run --rm --no-deps k6     # k6: 50 users for 30 s against POST /predict
docker compose up -d --scale api=1 --no-recreate            # try with 1 copy, then --scale api=3 again
```

- [nginx/nginx.conf](nginx/nginx.conf) spreads requests round-robin over the copies (`X-Upstream` response header
  shows which one answered) and re-reads Docker's DNS, so scaling up or down needs no Nginx restart.
- [loadtest/predict.js](loadtest/predict.js) sends random but valid passengers (no dataset rows); it fails if more
  than 1% of requests fail or p95 latency is above 500 ms.
- Each API copy uses one PyTorch thread, so more copies = more CPU used in parallel.

Measured on a MacBook Pro M1 (Docker Desktop, 10 CPUs), 50 virtual users, 30 s:

| API copies | Requests/s | Avg latency | p95 latency | Errors |
|---|---|---|---|---|
| 1 | 356 | 140 ms | 182 ms | 0 |
| 3 | 856 (2.4×) | 58 ms | 137 ms | 0 |

Lesson learned: `docker compose run k6` without `--no-deps` restarts the API with `replicas: 3`, so a "1 copy" test
silently ran with 3 copies and showed no difference.

## Status

- [x] Step 1 – Environment and repository skeleton
- [x] Step 2 – Data preparation and traffic simulator
- [x] Step 3 – Model training and MLflow
- [x] Step 4 – Prediction API
- [x] Step 5 – Scaling and load test
- [ ] Step 6 – Prediction logging and monitoring
- [ ] Step 7 – Drift detection and retraining
- [ ] Step 8 – CI/CD
- [ ] Step 9 – Demo scenario and documentation
- [ ] Step 10 – Presentation

## Quick check

```bash
docker compose up
# expected output: SkySat: Docker ortami hazir
```

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

## Status

- [x] Step 1 – Environment and repository skeleton
- [x] Step 2 – Data preparation and traffic simulator
- [ ] Step 3 – Model training and MLflow
- [ ] Step 4 – Prediction API
- [ ] Step 5 – Scaling and load test
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

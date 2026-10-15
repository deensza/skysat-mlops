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

## Training on Kaggle

The model is a PyTorch MLP ([training/train_torch.py](training/train_torch.py), adapted data handling from
[Ali Hassan's Apache 2.0 baseline](https://www.kaggle.com/code/alikalwar/airline-satisfaction-lightgbm-baseline-auc-0-9578)).
One-time: `pip install kaggle`, add your Kaggle API credentials, join the competition and accept its rules.

```bash
export KAGGLE_USERNAME=<your kaggle username>
make kaggle-train   # pushes a private notebook, waits for it, downloads artifacts/model.pt + submission.csv
```

`model.pt` is a single file (weights + preprocessing), loaded with `load_bundle()` and `predict_proba()` from the same script.
The competition data stays on Kaggle; the notebook is private.

## Status

- [x] Step 1 – Environment and repository skeleton
- [ ] Step 2 – Data preparation and traffic simulator
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

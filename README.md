# SkySat – MLOps Term Project

SE4458 Software Architecture & Design of Modern Large Scale Systems – Assignment 1 (Demo)
Topic: **MLOps – Build a simple ML model and demonstrate its complete lifecycle from training to production**

Team: Bermal Deniz Akkaya, Osman Şahin Güler

SkySat predicts whether an airline passenger is satisfied and shows the full MLOps lifecycle around that model:
data → training → experiment tracking & model registry (MLflow) → packaging (Docker) → serving (FastAPI) →
scaling & load testing (Nginx, k6) → monitoring (PostgreSQL, Grafana) → drift detection → retraining → CI/CD (GitHub Actions).

Dataset: [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
(the original data behind Kaggle Playground Series S6E10, *Predicting Airline Satisfaction*).

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

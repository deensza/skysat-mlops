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
- This repository is public, which the rules allow: they forbid *private* sharing outside the team, not public sharing.
  Code that is shared publicly must also be shared on the competition's Kaggle discussion forum or notebooks:
  post a link to this repo in the competition's Discussion tab. The data and trained models are never in the repository.
- Code: [MIT License](LICENSE) (an OSI-approved licence that allows commercial use, as the rules require for shared code).

## Project structure

| Folder | Purpose |
|---|---|
| `data/` | Dataset files (downloaded, not committed) |
| `docs/` | [Architecture](docs/architecture.md): how all the pieces fit together — start here |
| `simulator/` | Sends passenger records to the API (normal and drift modes) |
| `training/` | Training and retraining scripts, MLflow logging |
| `api/` | FastAPI prediction service |
| `monitoring/` | Grafana dashboards and database setup |
| `loadtest/` | k6 load test scripts |
| `nginx/` | Load balancer configuration |
| `tests/` | Automated tests (pytest) |
| `.github/workflows/` | CI/CD pipeline (GitHub Actions) |

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
- On a fresh start (`docker compose down -v`) there is no model yet: `/health` and `/predict` answer 503 until
  `python -m training.track` registers a champion; the API copies then load it by themselves.
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

## Step 6 – Prediction logging and monitoring

```bash
docker compose up -d --build        # adds PostgreSQL and Grafana
python -m simulator.simulate generate --mode normal --n 2000 --url http://localhost:8080/predict
open http://localhost:3000          # Grafana dashboard "SkySat – model in production" (no login needed)
```

- Every `POST /predict` is written to PostgreSQL (table `predictions`, [monitoring/init.sql](monitoring/init.sql)):
  the 22 inputs, probability, prediction, model version, which API copy answered, and the time it took.
- The simulator then sends the passenger's real answer (the survey result) to `POST /feedback`, which fills in
  `actual`. This is how the model's accuracy can be measured in production.
- If the database is down, the API still answers; it only logs a warning (monitoring must not break serving).
- Grafana reads PostgreSQL directly (no Prometheus). The dashboard is in the repo
  ([monitoring/grafana/dashboards/skysat.json](monitoring/grafana/dashboards/skysat.json)) and loads automatically:
  serving model version, number of predictions, accuracy and ROC AUC (overall and over time), requests per API copy,
  latency p50/p95, predicted vs real satisfied share, model version over time.
- ROC AUC is computed in SQL from ranks (Mann–Whitney) with average ranks for ties; checked against scikit-learn
  on the same rows (0.995575 both).

## Step 7 – Drift detection and retraining

```bash
python -m training.retrain --check-only   # only measure drift
python -m training.retrain                # measure drift; if found: retrain, compare, promote
```

[training/retrain.py](training/retrain.py) reads the latest 2,000 predictions from PostgreSQL and:

1. **Detects drift** with the Population Stability Index of every feature against the training data
   ([skysat/monitoring.py](skysat/monitoring.py)). Drift = any feature with PSI ≥ 0.2. No drift → nothing happens.
2. **Retrains** on the training data plus the recent passengers that have a real answer (70%, repeated 5 times so
   ~1,400 new rows are not drowned by 83,000 old ones). The other 30% is a holdout.
3. **Quality gate:** the champion and the challenger are scored on the **same** recent holdout. The challenger is
   registered if ROC AUC ≥ 0.95 and becomes `champion` only if it is better. The API copies load it within ~10 s.

Everything is logged to MLflow (`purpose=retraining`): PSI per feature, both scores, the challenger's score on the
original validation set, and the model file.

Measured end to end with `docker compose up` (3 API copies), 2,000 passengers per phase:

| Phase | Serving version | Accuracy (real answers) |
|---|---|---|
| Normal traffic | 1 | 96.25% |
| Winter storm, before retraining | 1 | 91.20% |
| Winter storm, after retraining (new passengers, `--seed 2`) | 2 | 94.35% |

- Normal traffic: highest PSI 0.001 → no drift. Storm traffic: PSI 0.44–1.30 on exactly the 9 features the storm
  changes (2 delays, 7 ratings), ≤ 0.004 on the other 13.
- Retraining: champion 0.957 vs challenger 0.990 ROC AUC on the same holdout → version 2 promoted; it still scores
  0.994 on the original (no storm) validation set.
- The weight 5 was picked in an offline test on 3,000 unseen storm passengers (weights 1 / 5 / 10 / 20 →
  ROC AUC 0.977 / 0.986 / 0.986 / 0.985, champion 0.954).
- Use a different `--seed` for traffic after retraining: the same seed sends the same passengers the model has just
  been trained on, and the recovery would look better than it is.

## Step 8 – CI/CD and continuous training (CT)

The map of all the pieces is in [docs/architecture.md](docs/architecture.md). Three automated loops:

| Loop | Workflow | Runs on | When | What |
|---|---|---|---|---|
| **CI** | `ci.yml` → `test`, `image` | GitHub's servers | every push / pull request | pytest on **fake data** (incl. training all 5 model types and the CT queue), compose check, API image build |
| **CD** | `ci.yml` → `deploy` | team PC (self-hosted runner) | push to `main`, after CI | push image to `ghcr.io/<repo>-api`, pull it on the PC, `docker compose up -d`, smoke test |
| **CT** | `ct.yml` → `candidates` | team PC | Mondays, by hand, new training code on `main` | (re)start the CT worker, queue the 5 candidates |
| **CT** | `ct.yml` → `drift` | team PC | daily, by hand | step 7's drift check; retrains the champion's model type only if drift |

**Why the team PC:** the competition rules forbid putting the Kaggle data on anyone else's servers, so training
on real data runs where the data already is. The `ct-worker` container mounts `data/`; the data is never copied
into an image, uploaded or logged.

### CT = a background worker + a queue

Training is its own service. Jobs wait in a queue (table `ct_jobs` in the PostgreSQL that already runs); the
`ct-worker` container takes the oldest one (`SELECT … FOR UPDATE SKIP LOCKED`, so no job is trained twice),
trains it, logs it to MLflow (`purpose=ct`, `ct_job`, `model_type`), applies the quality gate and makes it
`champion` if it beats the current one. Anyone can add jobs: you, the CT workflow, a cron job.

```bash
make start-ct                                   # start the worker in the background (needs `make data`)
make add-ct ARGS="--model random_forest"        # queue one model (mlp, logistic_regression, random_forest, gradient_boosting)
make add-ct ARGS="--model mlp --rating-one-hot --epochs 30 --name mlp-30-epochs"
make add-ct ARGS=--all                          # the 5 standard candidates
make list-ct                                    # queued / running / done / failed, with ROC AUC and version
docker compose --profile ct logs -f ct-worker   # watch it live
make stop-ct                                    # stop; a job in progress goes back to the queue
make smoke                                      # what CD checks after a deploy: /health + /predict via Nginx
```

The API serves scikit-learn models (`model.joblib`) as well as the MLP (`model.pt`); `GET /model` shows
`model_type`. A failing job is marked `failed` with the reason and the worker carries on.

Measured on the team PC (Linux, CPU only), `make add-ct ARGS=--all` on a fresh MLflow, original dataset
(83,123 train / 20,781 validation rows):

| Job | Model | Validation ROC AUC | Result |
|---|---|---|---|
| 1 | logistic_regression (one-hot ratings) | 0.9807 | version 1, champion |
| 2 | random_forest | 0.9941 | version 2, champion |
| 3 | gradient_boosting | 0.9952 | version 3, champion |
| 4 | mlp | 0.9948 | version 4, champion stays |
| 5 | mlp_rating_one_hot | 0.9955 | version 5, **champion** |

- The top three are within 0.0007 of each other: about the size of the seed-to-seed noise measured earlier
  (≤ 0.0005), so "best" here is partly luck.
- Checked: stopping the worker mid-job puts the job back to `queued`, and it is trained after `make start-ct`;
  a job with a bad parameter ends `failed` with the reason and the worker keeps running; `make add-ct` rejects
  unknown model or feature names before they reach the queue.

### One-time setup of the self-hosted runner (repo admin)

1. **Protect the PC from strangers' pull requests first.** The repo is public, and GitHub warns that on a public
   repo anyone can open a pull request whose (edited) workflow runs on a self-hosted runner. Our workflows never
   use the PC for pull requests, but a pull request can change that, so: Settings → Actions → General →
   *Fork pull request workflows from outside collaborators* → **Require approval for all external
   contributors**, and never approve a run from someone outside the team.
2. Settings → Actions → Runners → **New self-hosted runner** (Linux x64). Follow the commands, and when asked for
   labels add `skysat`. Install it as a service (`sudo ./svc.sh install && sudo ./svc.sh start`) so it survives
   reboots.
3. In the runner folder, create `.env` with the absolute path of the Kaggle files, then restart the service:
   ```
   SKYSAT_DATA_DIR=/home/<you>/Projects/skysat-mlops/data
   ```
4. The runner user needs Docker (`docker ps` works without sudo).

The runner shares the Docker Compose project `skysat-mlops` with your normal `docker compose up`, so CD and CT
work on the same MLflow models and prediction log you see locally.

## Status

- [x] Step 1 – Environment and repository skeleton
- [x] Step 2 – Data preparation and traffic simulator
- [x] Step 3 – Model training and MLflow
- [x] Step 4 – Prediction API
- [x] Step 5 – Scaling and load test
- [x] Step 6 – Prediction logging and monitoring
- [x] Step 7 – Drift detection and retraining
- [ ] Step 8 – CI/CD (code done; waiting for the self-hosted runner, see step 8)
- [ ] Step 9 – Demo scenario and documentation
- [ ] Step 10 – Presentation

## Quick check

```bash
docker compose up
# expected output: SkySat: Docker ortami hazir
```

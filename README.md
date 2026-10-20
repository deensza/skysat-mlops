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

The team uses Linux and macOS (Apple Silicon). You need Git, Docker (Docker Desktop on a Mac), `make` and
[uv](https://docs.astral.sh/uv/):

- macOS: `xcode-select --install` (gives git and make), then `brew install uv`
- Linux: git, make and uv from your package manager (or uv's install script from its website)
- Windows: not tested; use WSL2

```bash
make install          # .venv with Python 3.12 + the pinned packages (uv downloads Python 3.12 if needed)
make install-train    # + PyTorch, the Kaggle CLI and psycopg (training, retraining, competition track)
make test             # should end with "passed"
```

`make` uses `.venv` by itself; `source .venv/bin/activate` only if you want to run `python` directly.
On Linux without an NVIDIA GPU, PyPI's PyTorch brings ~2 GB of CUDA libraries; the comment at the top of
[requirements-train.txt](requirements-train.txt) shows the smaller CPU-only install. On a Mac PyPI's build is the
right one.

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
- **Quality gate:** a model is registered as `skysat-satisfaction` only if validation **ROC AUC ≥ 0.95**.
  Real models on this dataset score 0.98–0.996 (step 8), so the gate catches broken models, not merely weaker ones.
- The `champion` alias moves to the new version only if its ROC AUC is higher than the current champion's,
  **both measured on the same validation rows**: the champion is downloaded and re-scored, not judged by the number
  stored when it was trained (that number may come from other rows, e.g. step 7's recent holdout).
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
| **CI** | `ci.yml` → `test`, `image` | GitHub's servers | every push / pull request | pytest on **fake data** (incl. training all 5 model types), compose check, API image build |
| **CT** | `ct.yml` → `train` | GitHub's servers | `make add-ct`, the *Run workflow* button, Mondays, new training code on `main` | download the dataset, train, register, promote the best: all in the **shared MLflow** |
| **CD** | `ci.yml` → `deploy` | a team PC (self-hosted runner), optional | push to `main`, after CI | push image to `ghcr.io/<repo>-api`, pull it on the PC, `docker compose up -d`, smoke test |

Drift retraining (step 7, `python -m training.retrain`) stays a command you run where the stack is running,
because it needs that machine's prediction log; with the shared MLflow set in `.env` its model lands there too.

### CT on GitHub, results in one shared MLflow

No team PC has to be on. GitHub's run queue is the job queue and its machines are the workers:

```bash
make add-ct ARGS="--model random_forest"        # ask GitHub to train one model (mlp, logistic_regression, random_forest, gradient_boosting)
make add-ct ARGS="--model mlp --rating-one-hot --epochs 30 --name mlp-30-epochs"
make add-ct ARGS=--all                          # the 5 standard candidates
make list-ct                                    # recent CT runs: queued / in progress / success / failure
make ct-local ARGS="--model mlp --epochs 5"     # the same training on this computer (needs `make data`)
make smoke                                      # what CD checks after a deploy: /health + /predict via Nginx
```

Every CT run ([training/ct.py](training/ct.py)) downloads the **original** Kaggle dataset (the one `make data`
uses, not the competition data), trains each requested model on the same split, logs it to the team's MLflow on
**DagsHub** (`purpose=ct`, `model_type`), registers it if ROC AUC ≥ 0.95 and makes it `champion` if it beats the
current one. Both team members' APIs load the champion from that same MLflow, so everyone serves the same model.
The data only lives on GitHub's temporary machine; it is never printed, uploaded or committed.

The API serves scikit-learn models (`model.joblib`) as well as the MLP (`model.pt`); `GET /model` shows
`model_type`.

Measured on the team PC (Linux, CPU only) with the same training code, the 5 candidates on a fresh MLflow,
original dataset (83,123 train / 20,781 validation rows):

| Order | Model | Validation ROC AUC | Result |
|---|---|---|---|
| 1 | logistic_regression (one-hot ratings) | 0.9807 | version 1, champion |
| 2 | random_forest | 0.9941 | version 2, champion |
| 3 | gradient_boosting | 0.9952 | version 3, champion |
| 4 | mlp | 0.9948 | version 4, champion stays |
| 5 | mlp_rating_one_hot | 0.9955 | version 5, **champion** |

- The top three are within 0.0007 of each other: about the size of the seed-to-seed noise measured earlier
  (≤ 0.0005), so "best" here is partly luck.
- `make ct-local ARGS="--model logistic_regression --rating-one-hot"` reproduces 0.9807 exactly (fixed seeds).
- Before comparing, CT re-scores the champion on the same validation rows (`Champion (version 5) on the same rows:
  roc_auc 0.9955`). Known limitation: those rows are the original, pre-storm passengers, so after a drift
  retraining (step 7) a weekly CT round can still replace the storm-adapted model with one that is better on
  normal passengers. Check `GET /model` after a CT round if you just retrained for drift.

### One-time setup

**DagsHub (the shared MLflow).** Check DagsHub's pages as you go; its menus may have moved.

1. Both team members create a free DagsHub account.
2. One of you creates a **private** DagsHub repository (e.g. `skysat-mlops`) and adds the other as a collaborator.
   Private matters: only collaborators can see the runs and models. The free plan's limits at the time of writing:
   2 collaborators and 100 tracked runs in a private project (a 5-model round uses 5).
3. Its MLflow address is `https://dagshub.com/<owner>/<repo>.mlflow`. Each of you creates a DagsHub token and puts
   the three `MLFLOW_TRACKING_*` lines from [.env.example](.env.example) into your own `.env`.
4. **First check:** `make data` then `make ct-local ARGS="--model logistic_regression --rating-one-hot"`. It must end
   with `registered as version 1` after `Version 1 is the new champion`. That proves DagsHub accepts runs, model
   versions and the `champion` alias (which the API needs). If the alias step fails, stop and ask.

**GitHub (repository owner, once):** Settings → Secrets and variables → Actions.

| Kind | Name | Value |
|---|---|---|
| Variable | `MLFLOW_TRACKING_URI` | `https://dagshub.com/<owner>/<repo>.mlflow` |
| Variable | `MLFLOW_TRACKING_USERNAME` | the DagsHub user whose token is below |
| Secret | `MLFLOW_TRACKING_PASSWORD` | that DagsHub token |

Secrets never appear in logs, and runs started by strangers' pull requests never receive them. GitHub only lets you
start a workflow by hand once its file is on `main`, so merge first; after that `make add-ct CT_REF=<branch>`
trains with another branch's code.

**Optional, CD to a PC (self-hosted runner).** Only for a PC that is on when you push to `main`.

1. On this public repo, first set Settings → Actions → General → *Fork pull request workflows from outside
   collaborators* → **Require approval for all external contributors**, and never approve runs from strangers:
   otherwise a stranger's pull request could edit a workflow to run code on that PC.
2. Settings → Actions → Runners → **New self-hosted runner** (Linux x64), label `skysat`, install it as a service
   (`sudo ./svc.sh install && sudo ./svc.sh start`). The runner user needs Docker without sudo.
3. Set the repository variable `DEPLOY_TO_RUNNER` to `true`. Without it the `deploy` job is skipped.

## Kaggle competition track

A separate track for the leaderboard of [Playground S6E10](https://www.kaggle.com/competitions/playground-series-s6e10)
(metric ROC AUC, 10 submissions per day, final deadline 2026-10-31 23:59 UTC). Separate because the competition
files have other columns (no "Inflight service", True/False target) and must stay on the team's PCs, so this never
runs on GitHub and never touches the API's model registry.

```bash
make submission                       # 5-fold CV + artifacts/kaggle/submission.csv (~1 min), logged to MLflow `skysat-kaggle`
make submit MSG="hgb, 127 leaves"     # upload it (needs a Kaggle token in ~/.kaggle/access_token: Kaggle → Settings → API)
make kaggle-scores                    # the team's submissions and their public leaderboard scores
```

[training/competition.py](training/competition.py): scikit-learn gradient boosting, 5-fold stratified
cross-validation on the 699,635 competition train rows; the submission averages the 5 fold models' predictions.
Settings chosen by CV ROC AUC (all runs in the file's docstring):

| Setting | CV ROC AUC |
|---|---|
| scikit-learn defaults | 0.95767 ± 0.00063 |
| + the original dataset as extra training rows (allowed external data) | 0.95727 (worse in all 5 folds) |
| learning rate 0.05, up to 1000 rounds, 127 leaves (**default**) | **0.95853 ± 0.00053** |

- Same settings → byte-identical submission (sha256 printed and logged), so a leaderboard score can always be traced
  back to its MLflow run.
- The public leaderboard scores only part of the test set; the private part decides. Choose by CV, not by chasing
  the public score.

## Status

- [x] Step 1 – Environment and repository skeleton
- [x] Step 2 – Data preparation and traffic simulator
- [x] Step 3 – Model training and MLflow
- [x] Step 4 – Prediction API
- [x] Step 5 – Scaling and load test
- [x] Step 6 – Prediction logging and monitoring
- [x] Step 7 – Drift detection and retraining
- [ ] Step 8 – CI/CD (code done; waiting for the DagsHub + GitHub setup, see step 8)
- [ ] Step 9 – Demo scenario and documentation
- [ ] Step 10 – Presentation

## Quick check

```bash
docker compose up
# expected output: SkySat: Docker ortami hazir
```

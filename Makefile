# Common commands for SkySat. Run `make help` to list them.
#
#   make install        one-time: create .venv with the pinned packages (same as CI)
#   make install-train  one-time: add PyTorch for training
#   make data           step 2: clean + split the Kaggle files in data/
#   make train          step 3a: train the model -> artifacts/model.pt
#   make compare        compare the MLP with logistic regression and gradient boosting
#   make graph          plot which features drive satisfaction -> artifacts/eda/*.png
#   make add-ct ARGS=.. step 8: train on GitHub, e.g. ARGS="--model random_forest" or ARGS=--all (5 models)
#   make list-ct        step 8: recent CT runs on GitHub (results: the shared MLflow)
#   make ct-local ARGS= step 8: the same training on this computer
#   make smoke          step 8: smoke-test the running stack (what CD runs after a deploy)
#   make submission     competition: 5-fold CV + artifacts/kaggle/submission.csv (stays on this PC)
#   make submit MSG=..  competition: upload it to Kaggle;  make kaggle-scores: your public scores
#   make d-up / d-down  start / stop the Docker stack (data is kept);  make d-ps: what is running
#   make d-build        rebuild the images from the current code and restart what changed
#   make d-logs         follow the logs (one service: make d-logs SERVICE=api)
#   make test           run the tests
#   make ci             run the same checks as the GitHub Actions workflow

# Python 3.12 is what CI uses; override if needed: make install PYTHON=3.13
PYTHON ?= 3.12
# Use the project's .venv when it exists, else whatever python3 is on the PATH
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
KAGGLE := $(if $(wildcard .venv/bin/kaggle),.venv/bin/kaggle,kaggle)
# Settings from .env (e.g. the team's shared MLflow, see .env.example) reach every command below
-include .env
export MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD

.PHONY: help install install-train data train compare graph add-ct list-ct ct-local submission submit kaggle-scores smoke d-up d-build d-down d-ps d-logs d-loadtest d-wipe test compose-check ci

help:
	@awk -F ':.*## ' '/^[a-z-]+:.*## /{printf "%-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## create .venv and install the pinned packages (what CI installs)
	uv venv --python $(PYTHON) .venv
	uv pip install --python .venv/bin/python -r requirements.txt pytest==8.3.3

install-train: ## add PyTorch, the Kaggle CLI and psycopg to .venv (training, retraining, competition)
	uv pip install --python .venv/bin/python -r requirements-train.txt

data: ## step 2: clean and split data/train.csv + test.csv into data/processed/
	$(PY) -m training.prepare_data

train: ## step 3a: train on data/processed, write artifacts/model.pt
	$(PY) -m training.train

compare: ## compare the MLP with logistic regression and gradient boosting (3 seeds)
	$(PY) -m training.compare

graph: ## plot feature weights and relationships from the competition zip -> artifacts/eda/
	$(PY) -m training.analysis.eda

# Continuous training (step 8) runs on GitHub's servers (.github/workflows/ct.yml); results in the shared MLflow.
# CT_REF: the branch whose code trains (the workflow must exist there): make add-ct CT_REF=ml ARGS=--all
CT_REF ?= main

add-ct: ## step 8: train on GitHub: make add-ct ARGS="--model random_forest"  (ARGS=--all: the 5 candidates)
	gh workflow run ct.yml --ref $(CT_REF) -f args="$(ARGS)"
	@echo "Queued on GitHub. Follow it: make list-ct   Results: the shared MLflow (runs, versions, champion)"

list-ct: ## step 8: recent CT runs on GitHub
	gh run list --workflow ct.yml --limit 10

ct-local: ## step 8: the same training on this computer (needs make data): make ct-local ARGS="--model mlp"
	$(PY) -m training.ct $(ARGS)

# Kaggle competition track (training/competition.py). The competition data stays on this PC: never on GitHub.
COMPETITION := playground-series-s6e10

submission: ## competition: 5-fold CV + artifacts/kaggle/submission.csv (needs data/playground-series-s6e10.zip)
	$(PY) -m training.competition $(ARGS)

submit: ## competition: upload the submission: make submit MSG="what changed" (needs your Kaggle token)
	@test -n "$(MSG)" || { echo 'Say what this submission is: make submit MSG="hgb, 127 leaves"'; exit 1; }
	$(KAGGLE) competitions submit -c $(COMPETITION) -f artifacts/kaggle/submission.csv -m "$(MSG)"

kaggle-scores: ## competition: your team's submissions and their public leaderboard scores
	$(KAGGLE) competitions submissions -c $(COMPETITION)

# Docker stack (docker-compose.yml): MLflow, PostgreSQL, 3 API copies, Nginx, Grafana
SERVICE ?=

d-up: ## docker: start everything in the background (after editing .env too: changed settings are applied)
	docker compose up -d

d-build: ## docker: rebuild images from the current code and restart what changed (after editing api/, skysat/, training/train.py)
	docker compose up -d --build

d-down: ## docker: stop everything cleanly; all data is kept (use this, not `docker compose kill`)
	docker compose down

d-ps: ## docker: what is running, and is it healthy
	docker compose ps

d-logs: ## docker: follow the logs, Ctrl+C to stop (one service: make d-logs SERVICE=api)
	docker compose logs -f --tail 50 $(SERVICE)

d-loadtest: ## docker: k6 load test, 50 users for 30 s against POST /predict (step 5)
	docker compose --profile loadtest run --rm --no-deps k6

d-wipe: ## docker: stop and DELETE the local data (prediction log, local MLflow, Grafana); asks first
	@printf "Delete ALL local Docker data: prediction log, local MLflow models, Grafana? [y/N] "; read answer; \
	if [ "$$answer" = y ]; then docker compose down -v; else echo "Nothing deleted."; fi

smoke: ## step 8: /health + /predict through Nginx on localhost:8080
	bash .github/scripts/smoke-test.sh http://localhost:8080

test: ## run pytest
	$(PY) -m pytest -v

compose-check: ## validate docker-compose.yml (all profiles)
	docker compose --profile loadtest config --quiet

ci: test compose-check ## same two checks as the GitHub Actions workflow

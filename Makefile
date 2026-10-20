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
#   make test           run the tests
#   make ci             run the same checks as the GitHub Actions workflow

# Python 3.12 is what CI uses; override if needed: make install PYTHON=3.13
PYTHON ?= 3.12
# Use the project's .venv when it exists, else whatever python3 is on the PATH
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
# Settings from .env (e.g. the team's shared MLflow, see .env.example) reach every command below
-include .env
export MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD

.PHONY: help install install-train data train compare graph add-ct list-ct ct-local smoke test compose-check ci

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/'

install: ## create .venv and install the pinned packages (what CI installs)
	uv venv --python $(PYTHON) .venv
	uv pip install --python .venv/bin/python -r requirements.txt pytest==8.3.3

install-train: ## add PyTorch to .venv (needed for train / its tests)
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

smoke: ## step 8: /health + /predict through Nginx on localhost:8080
	bash .github/scripts/smoke-test.sh http://localhost:8080

test: ## run pytest
	$(PY) -m pytest -v

compose-check: ## validate docker-compose.yml (all profiles)
	docker compose --profile loadtest config --quiet

ci: test compose-check ## same two checks as the GitHub Actions workflow

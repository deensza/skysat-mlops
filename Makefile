# Common commands for SkySat. Run `make help` to list them.
#
#   make install        one-time: create .venv with the pinned packages (same as CI)
#   make install-train  one-time: add PyTorch for training
#   make data           step 2: clean + split the Kaggle files in data/
#   make train          step 3a: train the model -> artifacts/model.pt
#   make compare        compare the MLP with logistic regression and gradient boosting
#   make graph          plot which features drive satisfaction -> artifacts/eda/*.png
#   make start-ct       step 8: start the CT worker in the background (trains every queued model)
#   make add-ct ARGS=.. step 8: queue a model, e.g. ARGS="--model random_forest" or ARGS=--all (5 models)
#   make list-ct        step 8: CT jobs and results;  make stop-ct: stop the worker
#   make smoke          step 8: smoke-test the running stack (what CD runs after a deploy)
#   make test           run the tests
#   make ci             run the same checks as the GitHub Actions workflow

# Python 3.12 is what CI uses; override if needed: make install PYTHON=3.13
PYTHON ?= 3.12
# Use the project's .venv when it exists, else whatever python3 is on the PATH
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: help install install-train data train compare graph start-ct stop-ct add-ct list-ct smoke test compose-check ci

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

# CT worker (step 8): a background service that trains every job in the CT queue (training/ct_worker.py)
CT := HOST_UID=$$(id -u) HOST_GID=$$(id -g) docker compose --profile ct

start-ct: ## step 8: start the CT worker in the background (also MLflow + PostgreSQL; needs `make data`)
	$(CT) up -d --build ct-worker

stop-ct: ## step 8: stop the CT worker (a job in progress goes back to the queue)
	$(CT) stop ct-worker

add-ct: ## step 8: queue a model: make add-ct ARGS="--model random_forest"  (ARGS=--all: the 5 candidates)
	$(CT) run --rm --build ct-worker python -m training.ct_queue add $(ARGS)

list-ct: ## step 8: latest CT jobs and their results
	$(CT) run --rm --build ct-worker python -m training.ct_queue list

smoke: ## step 8: /health + /predict through Nginx on localhost:8080
	bash .github/scripts/smoke-test.sh http://localhost:8080

test: ## run pytest
	$(PY) -m pytest -v

compose-check: ## validate docker-compose.yml (all profiles)
	docker compose --profile ct --profile loadtest config --quiet

ci: test compose-check ## same two checks as the GitHub Actions workflow

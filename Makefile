# Common commands for SkySat, the complaint box (docs/llm.md). Run `make help` to list them.
#
#   make install        one-time: .venv with the pinned packages for the tests (no GPU, no torch)
#   make test           run the tests;   make ci: the tests and the compose check (CI on GitHub checks more)
#   make llm-install    one-time: .venv-llm with CUDA PyTorch, transformers, peft (NVIDIA GPU)
#   make llm-up         the Docker stack: API, intake (POST /reviews + the queue), PostgreSQL, Nginx, Grafana, MLflow
#   make simulate-seed  simulated passengers: training reviews for the first model, into the queue
#   make llm-train      the trainer on this PC's GPU;  make train-remote: on the cheapest free RunPod GPU
#   make simulate       live passengers (SCENARIO=harsh: the drift), scored by the model as they arrive
#   make d-up / d-down  start / stop the Docker stack (the API, MLflow, PostgreSQL, Nginx, Grafana)

# Python 3.12 is what CI uses; override if needed: make install PYTHON=3.13
PYTHON ?= 3.12
# Use the project's .venv when it exists, else whatever python3 is on the PATH
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
# Settings from .env (e.g. the team's shared MLflow, see .env.example) reach every command below
-include .env
export MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD
export HF_HUB_OFFLINE

# The llm-train / llm-serve commands use a local MLflow (sqlite), so a trial never touches the
# team's registry, whatever .env says. SHARED=yes: use the MLflow from .env instead (a new champion there is picked
# up by the server). The docker targets are not affected: they read .env themselves.
SHARED ?= no
LLM_MLFLOW ?= sqlite:///artifacts/llm-runs/mlflow.db
LLM_ENV := $(if $(filter yes,$(SHARED)),,env -u MLFLOW_TRACKING_USERNAME -u MLFLOW_TRACKING_PASSWORD MLFLOW_TRACKING_URI=$(LLM_MLFLOW))

.PHONY: help install test compose-check ci smoke d-up d-build d-down d-ps d-logs d-loadtest d-wipe \
	llm-install llm-train llm-promote llm-serve llm-up llm-drift simulate-seed simulate queue-stats train-remote train-remote-dry

help:
	@awk -F ':.*## ' '/^[a-z-]+:.*## /{printf "%-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## create .venv with the pinned packages for the tests and the replay (what CI installs)
	uv venv --python $(PYTHON) .venv
	uv pip install --python .venv/bin/python -r requirements.txt pytest==8.3.3

test: ## run pytest
	$(PY) -m pytest -v

compose-check: ## validate docker-compose.yml (all profiles)
	docker compose --profile loadtest config --quiet

ci: test compose-check ## pytest and the compose check (CI on GitHub also checks the server compose file and Nginx, and builds the image)

smoke: ## /health + POST /score through Nginx on localhost:8080 (needs a champion adapter)
	bash .github/scripts/smoke-test.sh http://localhost:8080

# Docker stack (docker-compose.yml): the API (the LLM), MLflow, PostgreSQL, Nginx, Grafana
SERVICE ?=

d-up: ## docker: start everything in the background (after editing .env too: changed settings are applied)
	docker compose up -d

d-build: ## docker: rebuild the API image from the current code and restart what changed
	docker compose up -d --build

d-down: ## docker: stop everything cleanly; all data is kept (use this, not `docker compose kill`)
	docker compose down

d-ps: ## docker: what is running, and is it healthy
	docker compose ps

d-logs: ## docker: follow the logs, Ctrl+C to stop (one service: make d-logs SERVICE=api)
	docker compose logs -f --tail 50 $(SERVICE)

d-loadtest: ## docker: gentle k6 load test (1 user, 1 min) against POST /score
	docker compose --profile loadtest run --rm --no-deps k6

d-wipe: ## docker: stop and DELETE the local data (prediction log, local MLflow, model cache, Grafana); asks first
	@printf "Delete ALL local Docker data: prediction log, local MLflow models, the 3 GB base model, Grafana? [y/N] "; read answer; \
	if [ "$$answer" = y ]; then docker compose down -v; else echo "Nothing deleted."; fi

# The complaint-box LLM (llm/, docs/llm.md): simulator -> POST /reviews (intake) -> queue -> trainer (RunPod or this PC)
LLM_PY := .venv-llm/bin/python
URL ?= http://localhost:8080
QUEUE ?= http://localhost:8001
SCENARIO ?= none
FROM ?= 2019-01
MONTHS ?= 1
PER_MONTH ?= 100
N ?= 2000
BRANCH ?= $(shell git rev-parse --abbrev-ref HEAD)

llm-install: ## LLM: .venv-llm with CUDA PyTorch + transformers/peft (needs an NVIDIA GPU; ~5 GB)
	uv venv --python $(PYTHON) .venv-llm
	uv pip install --python $(LLM_PY) torch==2.14.1 numpy==2.1.3 pandas==2.2.3 "psycopg[binary]==3.3.6" -r requirements-llm.txt

simulate-seed: ## LLM: simulated passengers post N (2000) reviews flown <= 2018 into the queue: the first model's training data
	$(PY) -m simulator.replay --url $(QUEUE) --seed-train $(N)

simulate: ## LLM: live passengers: make simulate FROM=2019-01 MONTHS=1 PER_MONTH=100 SCENARIO=harsh (scored as they arrive)
	$(PY) -m simulator.replay --url $(QUEUE) --from $(FROM) --months $(MONTHS) --per-month $(PER_MONTH) --scenario $(SCENARIO)

queue-stats: ## LLM: what waits in the queue: make queue-stats QUEUE=http://localhost:8001
	@curl -s $(QUEUE)/queue/stats; echo

llm-train: ## LLM: worker 2 on this PC's GPU against QUEUE (local MLflow): make llm-train ARGS="--n 300 --keep-pending"
	@mkdir -p artifacts/llm-runs
	$(LLM_ENV) $(LLM_PY) -m llm.trainer --queue-url $(QUEUE) $(ARGS)

train-remote: ## LLM: worker 2 on the cheapest free RunPod GPU (needs RUNPOD_API_KEY, QUEUE_URL/USER/PASSWORD; BRANCH must be pushed)
	$(PY) -m llm.runpod_pod --wait --branch $(BRANCH) --trainer-args="$(ARGS)"

train-remote-dry: ## LLM: print the RunPod request train-remote would send (passwords hidden), start nothing
	$(PY) -m llm.runpod_pod --dry-run --branch $(BRANCH) --trainer-args="$(ARGS)"

llm-promote: ## LLM: make a registered version the champion (the release step after --no-promote): make llm-promote VERSION=2 SHARED=yes
	@test -n "$(VERSION)" || { echo "usage: make llm-promote VERSION=2 [SHARED=yes]"; exit 1; }
	$(LLM_ENV) $(LLM_PY) -m llm.promote $(VERSION)

# A local adapter (artifacts/llm/adapter, from make llm-train ARGS=--no-mlflow) is served when it exists, else the
# MLflow champion. CPU only, like the server: CUDA_VISIBLE_DEVICES is emptied.
LLM_ADAPTER_DIR ?= $(if $(wildcard artifacts/llm/adapter/skysat_llm.json),artifacts/llm/adapter,)
llm-serve: ## LLM: the API on http://localhost:8090 (POST /score, /docs), CPU, from .venv-llm, no database
	$(LLM_ENV) CUDA_VISIBLE_DEVICES= LLM_ADAPTER_DIR=$(LLM_ADAPTER_DIR) $(LLM_PY) -m uvicorn llm.serve:app --host 127.0.0.1 --port 8090

llm-up: ## LLM: build and start the whole Docker stack (API, intake, PostgreSQL, Nginx, Grafana, MLflow)
	docker compose up -d --build

llm-drift: ## LLM: drift check on the last 24 h of the server's log: make llm-drift URL=http://localhost:8080
	$(PY) -m llm.drift_check --url $(URL) $(ARGS)

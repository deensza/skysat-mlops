# Common commands for SkySat, the complaint box (docs/llm.md). Run `make help` to list them.
#
#   make install        one-time: .venv with the pinned packages for the tests (no GPU, no torch)
#   make test           run the tests;   make ci: the checks GitHub Actions runs
#   make llm-install    one-time: .venv-llm with CUDA PyTorch, transformers, peft (NVIDIA GPU)
#   make llm-cycle      the whole lifecycle on this PC: train v1, see it fail under drift, retrain v2
#   make d-up / d-down  start / stop the Docker stack (the API, MLflow, PostgreSQL, Nginx, Grafana)

# Python 3.12 is what CI uses; override if needed: make install PYTHON=3.13
PYTHON ?= 3.12
# Use the project's .venv when it exists, else whatever python3 is on the PATH
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
# Settings from .env (e.g. the team's shared MLflow, see .env.example) reach every command below
-include .env
export MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD
export HF_HUB_OFFLINE

# The llm-train / llm-eval / llm-cycle / llm-serve commands use a local MLflow (sqlite), so a trial never touches the
# team's registry, whatever .env says. SHARED=yes: use the MLflow from .env instead (a new champion there is picked
# up by the server). The docker targets are not affected: they read .env themselves.
SHARED ?= no
LLM_MLFLOW ?= sqlite:///artifacts/llm-runs/mlflow.db
LLM_ENV := $(if $(filter yes,$(SHARED)),,env -u MLFLOW_TRACKING_USERNAME -u MLFLOW_TRACKING_PASSWORD MLFLOW_TRACKING_URI=$(LLM_MLFLOW))

.PHONY: help install test compose-check ci smoke d-up d-build d-down d-ps d-logs d-loadtest d-wipe \
	llm-install llm-train llm-eval llm-serve llm-up llm-replay llm-drift llm-cycle

help:
	@awk -F ':.*## ' '/^[a-z-]+:.*## /{printf "%-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## create .venv with the pinned packages for the tests and the replay (what CI installs)
	uv venv --python $(PYTHON) .venv
	uv pip install --python .venv/bin/python -r requirements.txt pytest==8.3.3

test: ## run pytest
	$(PY) -m pytest -v

compose-check: ## validate docker-compose.yml (all profiles)
	docker compose --profile loadtest config --quiet

ci: test compose-check ## the same two checks as the GitHub Actions workflow

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

# The complaint-box LLM (llm/, docs/llm.md): LoRA on a GPU, here (.venv-llm) or on Colab (notebooks/)
LLM_PY := .venv-llm/bin/python
LLM_SPLIT ?= train
SCENARIO ?= none
SCENARIO_ARG := $(if $(filter-out none,$(SCENARIO)),--scenario $(SCENARIO),)
URL ?= http://localhost:8080
FROM ?= 2019-01
MONTHS ?= 3
PER_MONTH ?= 50

llm-install: ## LLM: .venv-llm with CUDA PyTorch + transformers/peft (needs an NVIDIA GPU; ~5 GB)
	uv venv --python $(PYTHON) .venv-llm
	uv pip install --python $(LLM_PY) torch==2.14.1 numpy==2.1.3 pandas==2.2.3 -r requirements-llm.txt

llm-train: ## LLM: LoRA-tune, gate, log, register: make llm-train LLM_SPLIT=live-2019 SCENARIO=harsh (ARGS=--no-mlflow: log nothing)
	@mkdir -p artifacts/llm-runs
	$(LLM_ENV) $(LLM_PY) -m llm.train_lora --split $(LLM_SPLIT) $(SCENARIO_ARG) $(ARGS)

llm-eval: ## LLM: score the champion on LLM_SPLIT (and SCENARIO): make llm-eval LLM_SPLIT=live-2019 SCENARIO=harsh
	@mkdir -p artifacts/llm-runs
	$(LLM_ENV) $(LLM_PY) -m llm.train_lora --eval-only --split $(LLM_SPLIT) $(SCENARIO_ARG) $(ARGS)

# A local adapter (artifacts/llm/adapter, from make llm-train ARGS=--no-mlflow) is served when it exists, else the
# MLflow champion. CPU only, like the server: CUDA_VISIBLE_DEVICES is emptied.
LLM_ADAPTER_DIR ?= $(if $(wildcard artifacts/llm/adapter/skysat_llm.json),artifacts/llm/adapter,)
llm-serve: ## LLM: the API on http://localhost:8090 (POST /score, /docs), CPU, from .venv-llm, no database
	$(LLM_ENV) CUDA_VISIBLE_DEVICES= LLM_ADAPTER_DIR=$(LLM_ADAPTER_DIR) $(LLM_PY) -m uvicorn llm.serve:app --host 127.0.0.1 --port 8090

llm-up: ## LLM: build and start the whole Docker stack (same as d-up with a rebuild; the API is at :8080/score)
	docker compose up -d --build

llm-replay: ## LLM: send live reviews with their true labels: make llm-replay URL=.. SCENARIO=harsh FROM=2019-01 MONTHS=3 PER_MONTH=50
	$(PY) -m simulator.replay --url $(URL)/score --scenario $(SCENARIO) --from $(FROM) --months $(MONTHS) --per-month $(PER_MONTH)

llm-drift: ## LLM: drift check on the last 24 h of the server's log: make llm-drift URL=http://localhost:8080
	$(PY) -m llm.drift_check --url $(URL) $(ARGS)

llm-cycle: ## LLM: train v1 on <= 2018, score it on harsh 2019 (fails the gate), retrain v2 on harsh 2019 (promoted)
	@mkdir -p artifacts/llm-runs
	$(LLM_ENV) $(LLM_PY) -m llm.train_lora --split train
	$(LLM_ENV) $(LLM_PY) -m llm.train_lora --eval-only --split live-2019 --scenario harsh
	$(LLM_ENV) $(LLM_PY) -m llm.train_lora --split live-2019 --scenario harsh

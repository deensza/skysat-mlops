# Common commands for SkySat. Run `make help` to list them.
#
#   make install        one-time: create .venv with the pinned packages (same as CI)
#   make install-train  one-time: add PyTorch for training
#   make data           step 2: clean + split the Kaggle files in data/
#   make train          step 3a: train the model -> artifacts/model.pt
#   make test           run the tests
#   make ci             run the same checks as the GitHub Actions workflow
#   make kaggle-train   train the competition model on Kaggle (separate leaderboard track)

# Python 3.12 is what CI uses; override if yours is named differently: make install PYTHON=python3
PYTHON ?= python3.12
# Use the project's .venv when it exists, else whatever python3 is on the PATH
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: help install install-train data train test compose-check ci kaggle-train

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/'

install: ## create .venv and install the pinned packages (what CI installs)
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -r requirements.txt pytest==8.3.3

install-train: ## add PyTorch to .venv (needed for train / its tests)
	.venv/bin/pip install -r requirements-train.txt

data: ## step 2: clean and split data/train.csv + test.csv into data/processed/
	$(PY) -m training.prepare_data

train: ## step 3a: train on data/processed, write artifacts/model.pt
	$(PY) -m training.train

test: ## run pytest
	$(PY) -m pytest -v

compose-check: ## validate docker-compose.yml
	docker compose config --quiet

ci: test compose-check ## same two checks as the GitHub Actions workflow

# Train on Kaggle: make kaggle-train
# One-time setup: pip install kaggle, add your Kaggle API credentials, join the competition
# (accept its rules), and export KAGGLE_USERNAME=<your kaggle username>.
# Result: artifacts/kaggle/model.pt and submission.csv (gitignored). This is the competition
# track (competition CSVs, training/train_torch.py), separate from `make train`.

KAGGLE_USER ?= $(KAGGLE_USERNAME)
KERNEL_SLUG ?= skysat-train
KERNEL_ID   := $(KAGGLE_USER)/$(KERNEL_SLUG)
BUILD_DIR   := kaggle/build
OUT_DIR     := artifacts/kaggle
POLL_SECONDS ?= 20

kaggle-train: ## train the competition model on Kaggle (private notebook), download the result
	@command -v kaggle >/dev/null || { echo "kaggle CLI not found: pip install kaggle"; exit 1; }
	@test -n "$(KAGGLE_USER)" || { echo "set KAGGLE_USERNAME=<your kaggle username>"; exit 1; }
	rm -rf $(BUILD_DIR) && mkdir -p $(BUILD_DIR) $(OUT_DIR)
	cp training/train_torch.py $(BUILD_DIR)/
	sed "s/__USER__/$(KAGGLE_USER)/g; s/__SLUG__/$(KERNEL_SLUG)/g" kaggle/kernel-metadata.template.json > $(BUILD_DIR)/kernel-metadata.json
	kaggle kernels push -p $(BUILD_DIR)
	@echo "Waiting for the Kaggle run (private notebook $(KERNEL_ID))..."
	@while :; do \
	  s=$$(kaggle kernels status $(KERNEL_ID) 2>&1); echo "$$s"; \
	  case "$$(echo "$$s" | tr 'A-Z' 'a-z')" in \
	    *complete*) break ;; \
	    *error*|*fail*|*cancel*) echo "Kaggle run did not finish cleanly - see the notebook log on kaggle.com"; exit 1 ;; \
	  esac; \
	  sleep $(POLL_SECONDS); \
	done
	kaggle kernels output $(KERNEL_ID) -p $(OUT_DIR)
	@ls -l $(OUT_DIR)

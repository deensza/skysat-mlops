# Train on Kaggle: make kaggle-train
# One-time setup: pip install kaggle, add your Kaggle API credentials, join the competition
# (accept its rules), and export KAGGLE_USERNAME=<your kaggle username>.
# Result: artifacts/model.pt and artifacts/submission.csv (both gitignored).

KAGGLE_USER ?= $(KAGGLE_USERNAME)
KERNEL_SLUG ?= skysat-train
KERNEL_ID   := $(KAGGLE_USER)/$(KERNEL_SLUG)
BUILD_DIR   := kaggle/build
OUT_DIR     := artifacts
POLL_SECONDS ?= 20

.PHONY: kaggle-train
kaggle-train:
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

# The complaint box (LLM)

The only model of the project. A 1.5B-parameter LLM (Qwen2.5-1.5B-Instruct, Apache-2.0) is LoRA-tuned to read a **real
airline review** and answer with three lines. The lifecycle around it: gate, register in MLflow, serve, monitor,
detect drift, retrain.

```
Score: 7/10
Recommended: yes
Complaint: food
```

| | |
|---|---|
| Model | Qwen2.5-1.5B-Instruct, 3 GB, **frozen** + a LoRA adapter (r=16, ~0.3% of the weights, 17 MB) |
| Training | on a GPU: a local NVIDIA card (`.venv-llm`) or Colab T4 (`notebooks/train_lora.ipynb`). Never on GitHub |
| Registry | `skysat-reviews`, alias `champion`; a version is just the adapter |
| Gate | MAE <= 1.0, recommended >= 0.90, complaint >= 0.65 (`llm/gate.py`); a new adapter must also beat the champion on the same reviews |
| Serving | `llm/serve.py` on CPU, one review at a time, behind Nginx: `POST /score` |
| Monitoring | PostgreSQL table `llm_scores` -> Grafana; `GET /scores/recent` for the drift check |

## The data: real reviews, real labels

Kaggle `joelljungstrom/128k-airline-reviews` (licence DbCL-1.0; reviews scraped from airlinequality.com by the dataset's
author), the file `data/reviews/AirlineReviews.csv`. The labels are the reviewers' own: `OverallScore` (1-10),
`Recommended` (yes/no) and the seven sub-ratings (1-5; **0 means "not rated"**). The main complaint is the sub-rating(s)
tied for the lowest rated value if that value is <= 2 (seat, service, food, entertainment, ground service, wifi,
value), else `none`. A prediction is right if it is any of the tied names.

`llm/reviews.py` loads and cleans (review present, score 1-10, parseable flight date, recommended yes/no) and splits
with a fixed seed:

- **held-out airlines**: 20% of the airlines, never in training or validation;
- **train**: flown <= 2018, the other airlines; **validation**: a random 10% of those rows, not in train;
- **live**: flown >= 2019, all airlines, sorted by date: the traffic after training.

The prompt holds the airline, cabin, travel type, title and the review cut to 250 words. No date and no label.

**The dataset must never be committed, logged to MLflow, put in a Docker image or stored as text in Postgres.**
`data/` is in `.gitignore` and `.dockerignore`.

## The gate and what it measured

`llm/gate.py` parses the three lines and compares them with the labels: `score_mae`, `score_within1`,
`recommended_accuracy`, `complaint_accuracy`. A malformed field is wrong (error 9 for the score).
`combined = mean(1 - mae/9, recommended_accuracy, complaint_accuracy)` compares two adapters on the same reviews.
The limits (`MAE_MAX` 1.0, `RECOMMENDED_MIN` 0.90, `COMPLAINT_MIN` 0.65) come from the measured runs below.

| Adapter | Reviews | MAE | Recommended | Complaint | Gate |
|---|---|---|---|---|---|
| **v1** (2,000 reviews flown <= 2018; 5 min on an RTX 4070) | validation | 0.91 | 94.5% | 68% | passes, champion |
| v1 | real 2019-2022 traffic | 0.75-0.83 | | | passes: real drift does **not** hurt it |
| v1 | 2019 under the **harsh** survey scenario | 1.18 | 89.5% | | **fails** |
| **v2** (retrained on harsh 2019) | harsh 2019 | 0.41 | 95.5% | | passes, beats v1, **promoted automatically** |

Real reviews from 2019-2022 look like the training reviews to this model, so the drift is simulated:
`simulator/scenarios.py` changes the reviews and their true labels (`typos`, `short`, `html`, `harsh`, `mixed`).
`harsh` is a survey that rates everything lower, so the old adapter's scores are too high.

## Commands

```bash
kaggle datasets download joelljungstrom/128k-airline-reviews -p data/reviews --unzip   # once
make llm-install                      # once, on a machine with an NVIDIA GPU: .venv-llm (~5 GB)
make llm-train                        # v1: train on <= 2018, gate on validation, log, register, promote
make llm-eval LLM_SPLIT=live-2019 SCENARIO=harsh    # the champion on harsh 2019: fails the gate
make llm-train LLM_SPLIT=live-2019 SCENARIO=harsh   # retrain: v2, promoted if it beats the champion
make llm-cycle                        # those three, one after the other
make llm-train ARGS=--no-mlflow       # try it without logging anything (adapter in artifacts/llm/adapter)
```

The `llm-*` training commands use a local sqlite MLflow (`artifacts/llm-runs/mlflow.db`) unless you add `SHARED=yes`,
which uses the MLflow from `.env` (the team's). `HF_HUB_OFFLINE=1` is passed through.
`live-YYYY` trains on that year's live reviews and gates on a held-out 10% of the same year; with `--eval-only` it scores
the champion on the whole year. On Colab: open `notebooks/train_lora.ipynb`, pick a T4 GPU, add the secrets it lists.

MLflow gets parameters, metrics and the adapter. Never reviews, prompts or generated text.

## Serving

`llm/serve.py` loads the base model once (downloaded into `HF_HOME` the first time, ~3 GB) and the adapter
`skysat-reviews@champion` from MLflow on top of it. In Docker it is the `api` service (`mem_limit` 8 GB).

```bash
make d-up         # the Docker stack: api, MLflow, PostgreSQL, Nginx, Grafana (:8080 gateway, :3000 Grafana)
make llm-serve    # or locally on :8090 from .venv-llm, without a database
curl -s localhost:8080/score -H 'Content-Type: application/json' \
  -d '{"airline": "Example Air", "cabin": "Economy Class", "travel_type": "Solo Leisure", "title": "Cold food", "review": "..."}'
make llm-replay SCENARIO=harsh FROM=2019-01 MONTHS=3 PER_MONTH=50   # live reviews with their true labels
```

- `POST /score`: `airline`, `cabin`, `travel_type`, `title`, `review` (and optionally `true_score`,
  `true_recommended`) -> `score`, `recommended`, `complaint` (each `null` if the answer was malformed),
  `model_version`, `latency_ms`. Nginx routes only `/score`, `/model`, `/health`, `/docs` (and `/scores/recent`).
- **One review at a time** (a lock): one generation already uses every CPU core, and there is one model in memory.
- **A new champion is a new adapter, not a new model**: every `MODEL_RELOAD_SECONDS` (300) the alias is checked; a new
  version is loaded next to the old adapter on the same base, switched to, and the old one deleted. An adapter whose
  `skysat_llm.json` names another base model or prompt version (now 3) is refused and the old one keeps serving.
- `LLM_ADAPTER_DIR=<folder>` serves a local adapter without MLflow. `LLM_DTYPE`: `bfloat16` (default) or `float32`.

## Monitoring and drift check

With `DATABASE_URL`, each answer adds a row to `llm_scores` (`monitoring/init.sql`): `created_at`, `model_version`,
`latency_ms`, `score`, `recommended`, `complaint`, `true_score`, `true_recommended`. Never the review, the title or the
airline.

- **Grafana** (`http://localhost:3000`, dashboard `skysat`): requests per hour, latency p50/p95, live MAE and recommended
  accuracy per hour (where true labels were sent), complaint shares, model version over time.
- **`GET /scores/recent?hours=N`**: aggregates only: `count`, `mean_score`, `mae` and `recommended_accuracy` with the number
  of labelled requests behind each, `complaint_share`, `model_versions`. Never text.
- **`llm/drift_check.py`** (`make llm-drift URL=...`) reads that endpoint (basic auth from `API_USER` / `API_PASSWORD`)
  and says DRIFT if, on at least 50 labelled requests, the live MAE is above `gate.MAE_MAX` or the live recommended
  accuracy is below `gate.RECOMMENDED_MIN`. It prints a markdown table, always exits 0, and writes `drift=true|false`
  to `$GITHUB_OUTPUT`.

## GitHub Actions

| Workflow | What it does |
|---|---|
| `ci.yml` | tests (no torch), Docker Compose files valid, builds the API image (no model download); on `main` pushes it; deploys to the server if switched on, never for pull requests |
| `traffic.yml` | replays live reviews (scenario, month, reviews per month are inputs) to the server. Hourly schedule written but commented out: enable for the final run |
| `drift.yml` | runs the drift check; on drift opens the issue "Drift detected: retrain the LoRA adapter (make llm-cycle)". Schedule commented out the same way |

**Retraining does not run on GitHub**: there is no GPU. A person runs `make llm-cycle` (or `make llm-train LLM_SPLIT=live-2019
SCENARIO=harsh SHARED=yes`) on a GPU machine; the server then loads the new champion by itself.

## What changes for big models (and what does not)

- **The lifecycle is the same**: data, train, gate, register, promote, serve, monitor, retrain.
- **Weights are too big for the image and too big to retrain often.** The base model stays frozen and is downloaded
  once; each retrain produces a small adapter, so a "new model" is a small download.
- **Training needs a GPU**, so it is started by a person (or Colab) when the drift issue appears.
- **Serving needs RAM and is slow on CPU**: one review takes seconds, not milliseconds.
- **The gate is a behaviour test**, not one accuracy number: is the format right, and are the score, the recommendation
  and the complaint right on reviews the model has not seen?

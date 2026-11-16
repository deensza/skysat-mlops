# The complaint box (LLM)

The only model of the project. A 1.5B-parameter LLM (Qwen2.5-1.5B-Instruct, Apache-2.0) is LoRA-tuned to read a **real
airline review** and answer with three lines:

```
Score: 7/10
Recommended: yes
Complaint: food
```

Passengers' reviews reach the server, wait in a queue, and a rented GPU retrains the model from that queue:

```
passenger / simulator ──► POST /reviews (llm/intake.py, server) ──► queue.reviews (PostgreSQL)
                                │ each new review, with its true labels                 │ POST /queue/claim, /queue/done
                                ▼                                                        ▼
                     POST /score (llm/serve.py) ──► llm_scores ──► Grafana, drift check   trainer (llm/trainer.py) on the
                                ▲                                                         cheapest free RunPod GPU, or this PC
                                └──── MLflow skysat-reviews@champion (checked every MODEL_RELOAD_SECONDS) ◄── gate, register, promote
```

| | |
|---|---|
| Model | Qwen2.5-1.5B-Instruct, 3 GB, **frozen** + a LoRA adapter (r=16, ~0.3% of the weights, 17 MB) |
| Data | the 128k airline reviews on this PC (`data/reviews/AirlineReviews.csv`), posted to the server by the simulator as passengers |
| Queue | `queue.reviews` in the server's PostgreSQL, in its own schema that Grafana's user cannot read |
| Training | `llm/trainer.py` on a RunPod pod (RTX 3070 8 GB, $0.13/h, first choice; the pod deletes itself) or a local NVIDIA card. Never on GitHub |
| Registry | `skysat-reviews`, alias `champion`; a version is just the adapter |
| Gate | MAE <= 1.0, recommended >= 0.90, complaint >= 0.65 (`llm/gate.py`); a new adapter must also beat the champion on the same reviews |
| Serving | `llm/serve.py` on CPU, one review at a time, behind Nginx: `POST /score` |
| Monitoring | PostgreSQL table `llm_scores` -> Grafana; `GET /scores/recent` for the drift check |

## The data: real reviews, real labels

Kaggle `joelljungstrom/128k-airline-reviews` (licence DbCL-1.0; reviews scraped from airlinequality.com by the dataset's
author), the file `data/reviews/AirlineReviews.csv` on this PC. The labels are the reviewers' own: `OverallScore`
(1-10), `Recommended` (yes/no) and the seven sub-ratings (1-5; **0 means "not rated"**). The main complaint is the
sub-rating(s) tied for the lowest rated value if that value is <= 2 (seat, service, food, entertainment, ground service,
wifi, value), else `none`. A prediction is right if it is any of the tied names.

The prompt holds the airline, cabin, travel type, title and the review cut to 250 words. No date and no label.

**The dataset is never committed, logged to MLflow, put in a Docker image, or written to the table Grafana reads.**
Its reviews reach the server only through `POST /reviews` and live there in the private queue.

## The passengers: the simulator

`simulator/replay.py` posts reviews from the CSV to `POST /reviews`, one by one, as passengers would, with the
reviewer's own score, recommendation and sub-ratings. `split_reviews` (fixed seed) decides what it sends when:

- `make simulate-seed N=2000`: reviews flown <= 2018 (not the 20% held-out airlines) with `"live": false`. They go into
  the queue for the first model and are never sent to the model as traffic.
- `make simulate FROM=2019-01 PER_MONTH=100`: reviews flown >= 2019 in date order. The intake queues each one and sends
  it to the model with its true labels: the live traffic on the dashboard. The next training batch is made of them.
- `SCENARIO=harsh` (or `typos`, `short`, `html`, `mixed`; `simulator/scenarios.py`) changes the text and the labels
  first. `harsh` is a new survey that scores everything 2 points lower: the drift the current model misses.

Each review has a fixed id (`kaggle128k:<row>[:scenario]`), so posting it twice adds it once.

## The queue and the trainer

`queue.reviews` (`llm/queue_db.py`): **pending** (received) -> **claimed** (in a training batch) -> **trained**.

- `POST /queue/claim` takes the most recently received pending reviews (the model should learn what passengers say
  now) with `FOR UPDATE SKIP LOCKED`, so two trainers never get the same review, plus `replay` already-trained ones so
  a new adapter does not forget what the last one learned.
- `POST /queue/done` with `ok: true` marks the batch trained; `ok: false` (the trainer failed) puts it back to pending.
  A claim older than 3 hours (a pod that died) is put back by the intake.

`llm/trainer.py` claims up to `--n` (2,000) new reviews and `--replay` (500) trained ones. **10% of the new reviews are kept
back for the gate** and never trained on (at most `--n-gate-max`, 200), and no training row shares its text with a gate
row. A fresh LoRA adapter is trained on the frozen base; `llm/train_lora.py run()` gates it, scores the current champion
on the same gate reviews, logs to MLflow, registers the version if it passes, and makes it the champion if its
`combined` beats the champion's. Fewer than `--min-new` (200) new reviews: nothing is trained.

## The gate and what it measured

`llm/gate.py` parses the three lines: `score_mae` (a malformed score counts as 9), `score_within1`,
`recommended_accuracy`, `complaint_accuracy`; `combined = mean(1 - mae/9, recommended_accuracy, complaint_accuracy)`.

| Adapter | Trained on | Gate reviews | MAE | Recommended | Complaint | Result |
|---|---|---|---|---|---|---|
| v1 (13 Nov, the split CSV) | 2,000 flown <= 2018, 5 min, RTX 4070 | 2018 validation | 0.91 | 94.5% | 68% | passes, champion |
| v1 (13 Nov) | | harsh 2019 | 1.18 | 89.5% | | **fails** |
| v2 (13 Nov) | harsh 2019 | harsh 2019 | 0.41 | 95.5% | | passes, beats v1 |
| **v1 from the queue** (16 Nov) | 1,800 of 2,000 seeded, 322 s, RTX 4070 Laptop | 200 kept back | 0.88 | 92.0% | 71.5% | passes, champion |

Real 2019-2022 reviews look like the training reviews to this model (MAE 0.75-0.83 in the 13 Nov runs), so the drift
in the demo is simulated with `harsh`.

## RunPod: the trainer's GPU

`llm/runpod_pod.py` (`make train-remote`) creates a pod through RunPod's REST API (`https://api.runpod.io/v2/pods`),
trying GPU types cheapest first. Community-cloud prices from RunPod's catalog on 2026-11-16, per hour: RTX 3070 8 GB
0.13, RTX A4000 16 GB 0.17, RTX 3080 Ti 12 GB 0.18, RTX A4500 20 GB 0.19, RTX 4000 Ada 20 GB 0.20, RTX 3090 24 GB 0.22.

The pod runs `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`, clones `--branch` of this repo from GitHub (**the code
must be pushed**), installs transformers/peft/mlflow-skinny and runs the trainer against `QUEUE_URL`. It **deletes
itself** when the trainer exits, worked or not, with the key RunPod gives each pod (the account's key is never sent to
it). Two more guards: a timer inside the pod deletes it after `--max-minutes` (90), and `--wait` follows the log from
here and deletes it at that deadline too. On an 8 GB card PyTorch logged one out-of-memory warning (training finished);
if a 3070 runs out, retry with `--gpu "NVIDIA RTX A4000"`.

## Commands

```bash
make llm-up                                   # Docker: api, intake, PostgreSQL, Nginx, Grafana, MLflow
make simulate-seed N=2000                     # the first model's training reviews, into the queue
make llm-train                                # the trainer on this PC (local sqlite MLflow): v1
make simulate FROM=2019-01 PER_MONTH=100      # normal live passengers: Grafana shows the healthy baseline
make simulate FROM=2019-02 PER_MONTH=300 SCENARIO=harsh   # the drift: live MAE crosses the line
make llm-drift URL=http://localhost:8080      # DRIFT on at least 50 labelled requests
make llm-train                                # or: make train-remote (RunPod); v2 from the newest reviews
make queue-stats                              # counts per status
make train-remote-dry                         # the RunPod request, passwords hidden; starts nothing
```

`make llm-train` uses a local sqlite MLflow (`artifacts/llm-runs/mlflow.db`) unless you add `SHARED=yes`.
`train-remote` passes `QUEUE_URL`, `QUEUE_USER`, `QUEUE_PASSWORD` (the gateway's `trainer` login) and the MLflow from your
environment (`.env`: the team's DagsHub) to the pod. With `AUTO_TRAIN=1` (and those variables plus `RUNPOD_API_KEY` in the
server's `.env`) the intake starts a trainer pod by itself when `TRAIN_MIN_PENDING` (500) reviews wait, at most once every
`TRAIN_EVERY_HOURS` (6).

## Serving

`llm/serve.py` loads the base model once (downloaded into `HF_HOME` the first time, ~3 GB) and the adapter
`skysat-reviews@champion` from MLflow on top of it. In Docker it is the `api` service (`mem_limit` 8 GB).

- `POST /score`: `airline`, `cabin`, `travel_type`, `title`, `review` (and optionally `true_score`,
  `true_recommended`) -> `score`, `recommended`, `complaint` (each `null` if the answer was malformed),
  `model_version`, `latency_ms`. 15-24 s per review on a laptop CPU.
- **One review at a time** (a lock): one generation already uses every CPU core, and there is one model in memory.
- **A new champion is a new adapter, not a new model**: every `MODEL_RELOAD_SECONDS` the alias is checked; a new version
  is loaded next to the old adapter on the same base, switched to, and the old one deleted. An adapter whose
  `skysat_llm.json` names another base model or prompt version (now 3) is refused and the old one keeps serving.
- `LLM_ADAPTER_DIR=<folder>` serves a local adapter without MLflow. `LLM_DTYPE`: `bfloat16` (default) or `float32`.

## Monitoring and drift check

Each answer adds a row to `llm_scores` (`monitoring/init.sql`): `created_at`, `model_version`, `latency_ms`, `score`,
`recommended`, `complaint`, `true_score`, `true_recommended`. Never the review, the title or the airline.

- **Grafana** (`http://localhost:3000`, dashboard `skysat`): requests per hour, latency, live MAE and recommended
  accuracy per hour, complaint shares, model version over time.
- **`GET /scores/recent?hours=N`**: aggregates only (counts, means, shares, model versions). Never text.
- **`llm/drift_check.py`** (`make llm-drift`, and `drift.yml` on GitHub) says DRIFT if, on at least 50 labelled
  requests, the live MAE is above `gate.MAE_MAX` or the live recommended accuracy is below `gate.RECOMMENDED_MIN`.

## What changes for big models (and what does not)

- **The lifecycle is the same**: data, train, gate, register, promote, serve, monitor, retrain.
- **Weights are too big for the image and too big to retrain often.** The base model stays frozen; each retrain
  produces a small adapter, so a "new model" is a small download.
- **Training needs a GPU, so it is rented by the minute**: the trainer pod exists only while it trains.
- **Serving needs RAM and is slow on CPU**: one review takes seconds, not milliseconds.
- **The gate is a behaviour test**: is the format right, and are the score, the recommendation and the complaint right
  on reviews the model has not seen?

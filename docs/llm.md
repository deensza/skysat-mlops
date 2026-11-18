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
- A live review is traffic first and training data after: it can be claimed only once the model has scored it
  (`scored_at`), so every live review is measured exactly once, by a model that never trained on it. `GET /queue/stats`
  gives `ready`, the pending reviews a trainer could take now; the Train workflow needs 300 of them. Before 17 Nov
  2026 a trainer could claim reviews the model had not scored yet, and those never showed up as live traffic.
- Reviews are scored oldest first. If the API fails on one review three times (too long for its 150 s limit, or
  refused), the intake leaves that review out of the live traffic (`given_up` in `GET /queue/stats`) so the ones behind
  it are scored; it can still be trained on. An API that is down or still loading (no connection, or 503) makes the
  intake wait and gives up on nothing.
- `POST /queue/done` with `ok: true` marks the batch trained; `ok: false` puts it back to pending. The trainer sends
  `true` only if a version was registered (not when it failed, the gate failed, or with `--no-mlflow`).
  A claim older than 3 hours (a pod that died) is put back by the intake.

`llm/trainer.py` claims up to `--n` (2,000) new reviews and `--replay` (500) trained ones. **10% of the new reviews are kept
back for the gate** and never trained on (at least `--n-gate-min`, 100; at most `--n-gate-max`, 200), and no training
row shares its text with a gate row. A fresh LoRA adapter is trained on the frozen base; `llm/train_lora.py run()` gates it, scores the current champion
on the same gate reviews, logs to MLflow, registers the version if it passes, and makes it the champion if its
`combined` beats the champion's. Fewer than `--min-new` (300) new reviews: nothing is trained.

## The gate and what it measured

`llm/gate.py` parses the three lines: `score_mae` (a malformed score counts as 9), `score_within1`,
`recommended_accuracy`, `complaint_accuracy`; `combined = mean(1 - mae/9, recommended_accuracy, complaint_accuracy)`.

| Adapter | Trained on | Gate reviews | MAE | Recommended | Complaint | Result |
|---|---|---|---|---|---|---|
| v1 (13 Nov, the split CSV) | 2,000 flown <= 2018, 5 min, RTX 4070 | 2018 validation | 0.91 | 94.5% | 68% | passes, champion |
| v1 (13 Nov) | | harsh 2019 | 1.18 | 89.5% | | **fails** |
| v2 (13 Nov) | harsh 2019 | harsh 2019 | 0.41 | 95.5% | | passes, beats v1 |
| **v1 from the queue** (16 Nov) | 1,800 of 2,000 seeded, 322 s, RTX 4070 Laptop | 200 kept back | 0.88 | 92.0% | 71.5% | passes, champion |

The row "v1 from the queue" is a local training run of 16 Nov 2026 (RTX 4070 Laptop, local MLflow). The version 1 that
the server serves is another run, the one of the first row, registered in the team's DagsHub MLflow on 13 Nov 2026
(run a371483b, started 12:42 UTC): 2,000 training reviews, 1 epoch, 200 evaluation reviews; `score_mae` 0.91,
`recommended_accuracy` 0.945, `complaint_accuracy` 0.67 (the row says 68%), combined 0.838.

Real 2019-2022 reviews look like the training reviews to this model (MAE 0.75-0.83 in the 13 Nov runs), so the drift
in the demo is simulated with `harsh`.

### Through the server's queue, 16 and 17 Nov 2026

The first rows above came from files on a PC. These came from the server's queue: 1,996 history reviews were seeded
(4 of 2,000 posts failed), 100 healthy and 298 harsh reviews arrived as live traffic and were scored by the champion
(version 1), and the trainer ran three times (runs 1 to 3 below). Where these were measured: runs 1 to 3 ran on a
Google Colab T4 started by hand, before the RunPod path had been tried. Colab is not a supported way to train and
there is no Colab code in this repository; the supported paths are a RunPod pod and a local NVIDIA card. After the
third run 300 more harsh reviews arrived and were scored by the new champion (version 4). Runs 4 and 5 ran on
RunPod through the Train workflow on GitHub: run 4 started by hand, run 5 started by the drift check
([what ran on RunPod](#what-ran-on-runpod-17-nov-2026)).

| Live on the server | Requests | MAE (± 1 standard error) | Recommended | Drift check |
|---|---|---|---|---|
| Champion v1: healthy reviews, flown 2019-01 | 100 | 0.950 ± 0.083 | 95.0% | no drift |
| Champion v1: harsh reviews, flown 2019-02 | 300 | 1.150 ± 0.061 | 89.0% | **DRIFT** (on MAE) |
| Champion v1: both | 400 | 1.100 ± 0.050 | 90.5% | **DRIFT** |
| Champion v4 (after run 3): harsh reviews, flown 2019-03, the first 165 of 300 | 165 | 0.551 ± 0.066 | 97.0% ± 1.3 | no drift |
| Champion v4, drift check on GitHub, 17 Nov 08:06 UTC: all its answers of the last 24 h | 267 | 0.820 ± 0.080 | 89.9% ± 1.8 | no drift |
| Champion v4, drift check on GitHub, 17 Nov 13:12 UTC, after 492 more reviews with a strong label shift were posted (those scored so far) | 511 | 1.209 ± 0.075 | 81.6% ± 1.7 | **DRIFT** (on both) |
| Champion v4, drift check on GitHub, 17 Nov 15:10 UTC: the check that started training run 5 | 591 | 1.230 ± 0.070 | 81.2% ± 1.6 | **DRIFT** (on both) |
| Champion v1 again (the alias was moved back by hand; version 6 had served 58 requests and was not judged), drift check on GitHub, 17 Nov 17:32 UTC | 71 | 1.845 ± 0.255 | 77.5% ± 5.0 | no drift: not enough labelled requests (71 < 100) |

| Training run | Trained on | Gate reviews | New adapter: MAE / recommended / complaint / combined | Champion on the same reviews (v1 for runs 1-3, v4 for run 4) | Result |
|---|---|---|---|---|---|
| 1 | 1,796 of the 1,996 history reviews, 2,279 s | 200 kept back | 0.97 / 90.5% / 67.5% / 0.824 | 0.89 / 92.5% / 66.5% / 0.830 | passes, registered as version 3; the champion stays (0.830 >= 0.824) |
| 2 | 298 of the 398 live reviews + 500 replayed history reviews, 1,003 s | 100 kept back | 1.07 / 87.0% / 67.0% / 0.807 | 1.18 / 85.0% / 68.0% / 0.800 | **fails the gate**, not registered; the batch went back to pending |
| 3 | 298 of the 398 live reviews, no replay, 3 epochs (447 steps), 1,153 s | 100 kept back | 0.60 / 93.0% / 72.0% / 0.861 | 1.13 / 90.0% / 72.0% / 0.831 | passes, registered as **version 4, the new champion**; the server loaded it 13 s later |
| 4 (17 Nov, RunPod RTX 3070, Train run 44030831433, started by hand) | a batch of 300 queue reviews less the gate reviews, no replay, 3 epochs; the pod existed about 5 minutes | 100 kept back | 0.33 / 93.0% / 68.0% / 0.858 | combined 0.861 (the other measures were not noted) | passes, registered as version 5; the champion stays version 4 (0.861 >= 0.858) |
| 5 (17 Nov, RunPod RTX 3080 Ti, started by the drift check, run 44111118456) | a batch of 405 queue reviews less the gate reviews, no replay, 3 epochs; the pod existed 5 min 32 s | 100 kept back (305 trained on) | 0.23 / 100.0% / 71.0% / 0.895 (within 1 point 0.94; DagsHub run a62a1127; 80 s of training) | combined 0.735 (v4; the other measures were not read) | registered as **version 6 and promoted**: the trainer registers only what passes the gate and promotes only what beats the champion on the same reviews. The server loaded it and served it until the alias was moved back to version 1 by hand (by 17:12 UTC) |

What run 2 shows. The retrained adapter beat the champion on the drifted reviews (0.807 against 0.800) but did not
reach the gate, so nothing was registered and version 1 went on serving. `harsh` changes the labels and leaves the
text as it is, and only about three quarters of the 298 new training reviews carried the new labels; the 500 replayed
ones and the healthy ones carried the old labels for the same kind of text. The v2 of 13 Nov, trained on harsh reviews only,
reached 0.41.

What run 3 shows. The same batch without the replayed reviews (`--replay 0 --epochs 3`, about as much training as
run 2: 447 steps against 399) passed the gate and beat the champion on the same 100 held-back reviews. What made the
difference was leaving out the reviews with the old labels. So the drift check starts its training with these
arguments (`drift.yml`); a training started by hand keeps the trainer's default, `--replay 500` and one epoch. Three
cautions:

- The 0.60 is on 100 held-back reviews of the same batch as the training reviews, and `harsh` pushes many scores down
  to 1, which makes them easier to predict than healthy ones.
- The live row above is the stronger evidence: 165 reviews of another month, never seen, scored by the server.
- Version 4 was measured on harsh reviews only. How it does on healthy ones (the old labels) was not measured; it has
  probably got worse there, because that is what the new labels ask for.

Runs 1 to 3 were timed on the Colab T4: 1,796 reviews took 2,279 s (322 s on an RTX 4070 Laptop) and PyTorch logged
one out-of-memory warning (training finished). Run 4's numbers are from its MLflow run. For runs 4 and 5 the pod's
own log could not be read ([below](#what-ran-on-runpod-17-nov-2026)), so the training time alone is not known, only
how long the pod existed.

## RunPod: the trainer's GPU

`llm/runpod_pod.py` (`make train-remote`) creates a pod through RunPod's REST API (`https://api.runpod.io/v2/pods`),
trying GPU types cheapest first. Community-cloud prices from RunPod's catalog on 2026-11-16, per hour: RTX 3070 8 GB
0.13, RTX A4000 16 GB 0.17, RTX 3080 Ti 12 GB 0.18, RTX A4500 20 GB 0.19, RTX 4000 Ada 20 GB 0.20, RTX 3090 24 GB 0.22,
and as a last resort RTX 4090 24 GB 0.34.

The pod runs `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`, clones `--branch` of this repo from GitHub (**the code
must be pushed**), installs transformers/peft/mlflow-skinny and runs the trainer against `QUEUE_URL`. It **deletes
itself** when the trainer exits, worked or not, with the key RunPod gives each pod (the account's key is never sent to
it). Two more guards: a timer inside the pod deletes it after `--max-minutes` (90), and `--wait` follows the pod from
here and deletes it at that deadline too. On an 8 GB card PyTorch logged one out-of-memory warning (training finished);
if a 3070 runs out, retry with `--gpu "NVIDIA RTX A4000"`.

`--wait` is written to print the pod's log and return the trainer's exit code from it. RunPod answers 403 to this
project's API key on the pod-log endpoint ("Pod log: HTTP 403", every run of 17 Nov 2026), so the launcher takes the
outcome from the server: it reads `/queue/stats` and `/model` before and after. More reviews marked trained means a
version was registered (exit code 0); a new serving version means it was promoted. Without the log a failed gate, too
few reviews and a crash look the same: nothing was registered, and the run ends as failed. The trainer's own lines
(gate numbers, training time) are then only in MLflow.

If no GPU type in the list is free, the launcher stops with "No GPU had capacity": nothing is rented and nothing is
billed. The Train workflow then fails without its "A GPU was rented" step, so the cooldown does not start and the next
drift check may try again. Every type in the list can be out of stock at once. On 17 Nov 2026 at 13:12 UTC all six
types then in the list answered HTTP 400 "There are no longer any instances available". The RTX 4090 ($0.34/h) was
added the same hour as a seventh type, tried last. At 13:38 and 14:24 UTC all seven, the RTX 4090 included, gave the
same answer. At 15:10 UTC the RTX 3070 and the RTX A4000 were still out of stock and an RTX 3080 Ti was free.

### What ran on RunPod, 17 Nov 2026

| Started by | Start (UTC) | GPU | The pod existed | Trainer arguments | What happened |
|---|---|---|---|---|---|
| Train run 44030831433, by hand | 06:23 | RTX 3070 | about 5 min | `--replay 0 --epochs 3` | training run 4 above: 300 reviews marked trained, version 5 registered, champion stays version 4 |
| Train run 44032765219, by hand | 06:39 | RTX 3070 | 3 min 48 s | `--min-new 100000` (a test of the launcher: the trainer trains nothing) | nothing registered, as intended |
| Train run 44033547732, by hand | 06:45 | RTX 3070 | 5 min 12 s | `--min-new 100000` | "reviews marked trained 2694 -> 2694, serving model 4 -> 4": nothing registered, as intended |
| Drift check 44090298125 (by hand), its retrain job | 13:12 | none | - | `--replay 0 --epochs 3` | DRIFT on version 4, Train called; "No GPU had capacity" for the six types then in the list; nothing rented |
| Drift check 44094574881 (by hand), its retrain job | 13:38 | none | - | `--replay 0 --epochs 3` | DRIFT on version 4, Train called; "No GPU had capacity" for all seven types; nothing rented |
| Drift check 44102632827 (by hand), its retrain job | 14:24 | none | - | `--replay 0 --epochs 3` | the same: all seven types out of stock; nothing rented |
| Drift check 44111118456 (by hand), its retrain job | 15:10 | RTX 3080 Ti, $0.18/h (RTX 3070 and RTX A4000 out of stock) | 5 min 32 s | `--replay 0 --epochs 3` | training run 5 above: DRIFT on version 4, Train called; "reviews marked trained 2694 -> 3099, serving model 4 -> 6"; version 6 registered and promoted; the run is green |

All three hand-started Train runs show as failed on GitHub. The first two ended before the launcher could ask the server
(added the same morning), so their outcome read as unknown; the third registered nothing, which is what its arguments
ask for. After each run the sweep found no pod ("Trainer pods left behind: 0").

Cost by RunPod's price list (the bill itself was not read): about 5 minutes on an RTX 3070 at $0.13/h is about one
cent per run; 5 min 32 s on the RTX 3080 Ti at $0.18/h is about 2 cents. The 45 minute limit at the dearest type in
the list (RTX 4090, $0.34/h) is the upper bound, about $0.26 per run.

The automatic chain on 17 Nov 2026: drift check -> DRIFT -> issue -> Train workflow -> cooldown and queue check ->
pod -> gate -> version 6 registered and promoted -> the server loaded it. It completed once, at the fourth call; the
first three found no free GPU. The drift checks were started by hand: GitHub's hourly schedule had not started one by 17:35 UTC that
day (two scheduled runs came later and failed: the server was off). Not in the log: version 6's gate numbers; they are in its DagsHub run (table above): recommended 100.0% is on 100
reviews only.
Version 6 was not judged on live traffic: it had served 58 requests (the check needs 100) when the alias was moved
back to version 1 by hand, by 17:12 UTC.

## Commands

```bash
make llm-up                                   # Docker: api, intake, PostgreSQL, Nginx, Grafana, MLflow
make simulate-seed N=2000                     # the first model's training reviews, into the queue
make llm-train LLM_MLFLOW=http://localhost:5050   # the trainer on this PC, into the stack's MLflow: v1, the API loads it
make simulate FROM=2019-01 PER_MONTH=100      # normal live passengers: Grafana shows the healthy baseline
make simulate FROM=2019-02 PER_MONTH=300 SCENARIO=harsh   # the drift: live MAE crosses the line
make llm-drift URL=http://localhost:8080      # DRIFT on at least 100 labelled requests
make queue-stats                              # counts per status; "ready": what a trainer could take now (300 needed)
make llm-train LLM_MLFLOW=http://localhost:5050 ARGS="--replay 0 --epochs 3"   # v2 from the new reviews only
make train-remote-dry                         # the RunPod request, passwords hidden; starts nothing
```

Plain `make llm-train` uses a local sqlite MLflow (`artifacts/llm-runs/mlflow.db`), which the API in Docker does not
read: use it for trials. `LLM_MLFLOW=http://localhost:5050` trains into the stack's own MLflow, `SHARED=yes` into the
team's (from `.env`). `make train-remote` runs the same trainer on RunPod against a queue the pod can reach (the
server's, not `localhost`). It passes `QUEUE_URL`, `QUEUE_USER`, `QUEUE_PASSWORD` (the gateway's `trainer` login) and the MLflow from your
environment (`.env`: the team's DagsHub) to the pod. With `AUTO_TRAIN=1` (and those variables plus `RUNPOD_API_KEY` in the
server's `.env`) the intake starts a trainer pod by itself when `TRAIN_MIN_PENDING` (500) reviews wait, at most once every
`TRAIN_EVERY_HOURS` (6).

## Serving

`llm/serve.py` loads the base model once (downloaded into `HF_HOME` the first time, ~3 GB) and the adapter
`skysat-reviews@champion` from MLflow on top of it. In Docker it is the `api` service (`mem_limit` 8 GB).

- `POST /score`: `airline`, `cabin`, `travel_type`, `title`, `review` (and optionally `true_score`,
  `true_recommended`) -> `score`, `recommended`, `complaint` (each `null` if the answer was malformed),
  `model_version`, `latency_ms`. 15-24 s per review on a laptop CPU.
- `POST /ask`: the review as plain text in the request's body (no JSON, no other fields) -> the same answer, with
  the review repeated in it. On the `/docs` page it is one text box with three ready-made reviews to pick from. It is
  not added to `llm_scores`, so trying the model out changes neither the dashboard nor the drift check.
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
- **`GET /scores/recent?hours=N`**: aggregates only (counts, means, shares, model versions). Never text. The optional
  `version=V` counts only the answers of that model version (`model_versions` still lists all of them).
- **`llm/drift_check.py`** (`make llm-drift`, and `drift.yml` on GitHub) says DRIFT if, on at least 100 labelled
  requests, the live MAE is above `gate.MAE_MAX` or the live recommended accuracy is below `gate.RECOMMENDED_MIN`,
  by more than one standard error of the live sample. It judges only the model that is serving now (it reads
  `GET /model` first), so a retrained model is not blamed for the old one's answers; `--all-versions` counts every
  version, as before. If the server cannot be read it exits with code 2.

## What changes for big models (and what does not)

- **The lifecycle is the same**: data, train, gate, register, promote, serve, monitor, retrain.
- **Weights are too big for the image and too big to retrain often.** The base model stays frozen; each retrain
  produces a small adapter, so a "new model" is a small download.
- **Training needs a GPU, so it is rented by the minute**: the trainer pod exists only while it trains.
- **Serving needs RAM and is slow on CPU**: one review takes seconds, not milliseconds.
- **The gate is a behaviour test**: is the format right, and are the score, the recommendation and the complaint right
  on reviews the model has not seen?

# Demo (3 minutes)

Recorded as a video beforehand (waiting moments cut) and narrated live, after the slides. The story: **passengers'
reviews arrive, the model degrades on them, the system notices, a GPU retrains it from the queue, the new model
goes live by itself.** The demo stack runs on one PC and trains on that PC's GPU. The rented RunPod GPU is shown from
the recorded run of 17 Nov 2026 on GitHub (step 9): a pod needs a queue address it can reach from the internet, and it
would write to the team's registry.

## Keep the demo away from the team's registry

The demo promotes new champions. `make llm-train` uses a local sqlite MLflow (`artifacts/llm-runs/mlflow.db`) unless you
pass `SHARED=yes`: **never pass `SHARED=yes` in the demo**. For Docker use a separate project and no shared MLflow:

```bash
make d-down                                   # stop your normal stack first (same ports)
export COMPOSE_PROJECT_NAME=skysat-demo       # own database and queue, own local MLflow, own model cache
alias dmake='make MLFLOW_TRACKING_URI= MLFLOW_TRACKING_USERNAME= MLFLOW_TRACKING_PASSWORD='
```

The API in Docker reads the champion from that project's MLflow service: train into it with
`LLM_MLFLOW=http://localhost:5050`. Set `MODEL_RELOAD_SECONDS=10` in `.env`, so a promotion shows within seconds.

## Before recording

```bash
dmake llm-up                                  # API, intake, MLflow, PostgreSQL, Nginx, Grafana (first start: 3 GB base model)
make simulate-seed N=2000                     # 2,000 reviews flown <= 2018 into the queue (training only, not scored)
make llm-train LLM_MLFLOW=http://localhost:5050   # v1 (~6 min on an RTX 4070): the champion
make simulate FROM=2019-01 PER_MONTH=60       # normal passengers: the healthy baseline in Grafana (scored as they arrive)
```

Wait until the baseline is scored (`make queue-stats`: `unscored` 0). A review takes ~20 s on a laptop CPU, so the 300
harsh reviews of step 4 take about 1 h 40 min to score: send them well before recording.
Open Grafana at http://localhost:3000 (time range **Last 1 hour**, refresh **5s**) and MLflow at http://localhost:5050.

## The demo

| # | Command / screen | What the audience sees |
|---|---|---|
| 1 | Architecture slide | passengers → `/reviews` → queue → trainer (RunPod on the server; this PC's GPU in the demo) → gate → MLflow → API → Grafana |
| 2 | Swagger `http://localhost:8001/docs` → `POST /reviews` | a passenger writes a review with their own score; it is queued, then scored by v1 |
| 3 | Grafana | live MAE / recommended accuracy healthy, model version 1 |
| 4 | `make simulate FROM=2019-02 PER_MONTH=300 SCENARIO=harsh` | a harsher survey: live MAE and accuracy cross the red lines as the reviews are scored |
| 5 | `make llm-drift URL=http://localhost:8080` | markdown table, `DRIFT` on at least 100 labelled requests |
| 6 | `make queue-stats` | the new reviews waiting in the queue. Wait until `ready` is 300 or more: a review can be trained on only after the model has scored it, and the trainer needs 300 new ones |
| 7 | `make llm-train LLM_MLFLOW=http://localhost:5050 ARGS="--replay 0 --epochs 3"` | this PC's GPU claims the newest batch, trains v2 on the new reviews only, the gate passes, v2 beats v1 on the same reviews, promoted |
| 8 | Grafana + `GET /model` | the API loads v2 by itself; accuracy recovers on the next harsh reviews |
| 9 | GitHub → Actions → Drift check, run 44111118456 of 17 Nov 2026 | the same loop on the server, with a rented GPU: the drift table says DRIFT; in the job `retrain / train`, "Pod ... created on NVIDIA GeForce RTX 3080 Ti", "Pod ... is gone" 5 min 32 s later, "serving model 4 -> 6", "Trainer pods left behind: 0". The GPU ($0.18/h) existed only while it trained |

- Step 7 takes about 5 minutes on an RTX 4070 (v1 took 322 s; time the retraining at the rehearsal); film it
  beforehand and cut the wait.
- Step 7 without `--replay 0 --epochs 3` uses the default, 500 replayed old reviews and one epoch: on the server that
  retraining failed the gate (16 Nov 2026). Without `LLM_MLFLOW=http://localhost:5050` the new version goes into a
  sqlite file the demo API does not read.
- Do not use `make train-remote` here: the pod cannot reach `localhost`, and it would register in the team's MLflow.
- Send a few more harsh reviews after step 7 (`make simulate FROM=2019-03 PER_MONTH=40 SCENARIO=harsh`) so v2 is
  scored on reviews it was not trained on.
- Real timings and numbers: [docs/llm.md](../docs/llm.md). Re-measure on the demo machine and note the date here.

## Afterwards: switch everything off

```bash
make d-wipe                                   # answer y: deletes ONLY the skysat-demo project
unset COMPOSE_PROJECT_NAME; unalias dmake     # or just close this terminal
```

The demo rents no pod. If you ran `make train-remote` against the server for another reason, check RunPod shows no pod
(the pod deletes itself; `make train-remote` also deletes it at its deadline).

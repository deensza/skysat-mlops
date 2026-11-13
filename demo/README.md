# Demo (3 minutes)

Recorded as a video beforehand (waiting moments cut) and narrated live, after the slides. The story: **the model degrades,
the system notices, one command retrains it, it recovers.**

## Keep the demo away from the team's registry

The demo promotes a new champion. The `llm-*` training commands already use a local sqlite MLflow
(`artifacts/llm-runs/mlflow.db`) unless you pass `SHARED=yes`: **never pass `SHARED=yes` in the demo**. For Docker use
a separate project and no shared MLflow settings:

```bash
make d-down                                   # stop your normal stack first (same ports)
export COMPOSE_PROJECT_NAME=skysat-demo       # own database, own local MLflow, own model cache
alias dmake='make MLFLOW_TRACKING_URI= MLFLOW_TRACKING_USERNAME= MLFLOW_TRACKING_PASSWORD='
```

The API in Docker reads the champion from the MLflow service of that project. To give it one, train into that MLflow:
`MLFLOW_TRACKING_URI=http://localhost:5050` for the training commands (set `LLM_MLFLOW=http://localhost:5050`).

## Before recording

```bash
make llm-install                              # once; the data in data/reviews/AirlineReviews.csv
dmake d-up                                    # API, MLflow, PostgreSQL, Nginx, Grafana (first start downloads the 3 GB base model)
make llm-train LLM_MLFLOW=http://localhost:5050                     # v1 (~5 min on an RTX 4070): the champion
dmake smoke                                   # one review through Nginx
dmake llm-replay MONTHS=1 PER_MONTH=100       # normal live traffic: Grafana shows the healthy baseline
```

Open Grafana at http://localhost:3000 (time range **Last 1 hour**, refresh **5s**) and MLflow at http://localhost:5050.

## The demo

| # | Command / screen | What the audience sees |
|---|---|---|
| 1 | MLflow → Models → `skysat-reviews` | v1 champion: MAE 0.91, recommended 94.5%, complaint 68% |
| 2 | `dmake smoke` | one review → score, recommended, complaint, model version, latency |
| 3 | `dmake llm-replay MONTHS=1 PER_MONTH=100` | normal 2019 traffic: live MAE 0.75-0.83 in Grafana (real drift does not hurt) |
| 4 | `dmake llm-replay SCENARIO=harsh MONTHS=1 PER_MONTH=100` | the harsh survey: live MAE and recommended accuracy cross the red lines |
| 5 | `dmake llm-drift` | markdown table, `DRIFT` on at least 50 labelled requests |
| 6 | `make llm-train LLM_SPLIT=live-2019 SCENARIO=harsh LLM_MLFLOW=...` | v2 (MAE 0.41, recommended 95.5%) beats v1 (harsh 2019: MAE 1.18, 89.5%) and is promoted |
| 7 | `dmake llm-replay SCENARIO=harsh FROM=2019-04 MONTHS=1 PER_MONTH=100` | the API loads v2 by itself (`MODEL_RELOAD_SECONDS`; set it to 10 in `.env` for the demo); Grafana: version 2, accuracy recovers |

- Step 6 is long (training); film it beforehand and cut the wait. `make llm-cycle` does v1, the failing score and v2 in one go.
- A different month in step 7 gives reviews v2 was not trained on.
- Real timings and numbers: [docs/llm.md](../docs/llm.md). Re-measure on the demo machine and note the date here.

## Afterwards

```bash
make d-wipe                                   # answer y: deletes ONLY the skysat-demo project
unset COMPOSE_PROJECT_NAME; unalias dmake     # or just close this terminal
```

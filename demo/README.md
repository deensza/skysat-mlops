# Step 9 – Demo (3 minutes)

Recorded as a video beforehand (waiting moments cut) and narrated live, after the slides. Three parts:

1. **Model selection** – CT trained 5 model types; the quality gate picked the champion (DagsHub).
2. **Kaggle** – the same approach on the competition data: 5-fold CV, a new submission, the leaderboard score.
3. **Drift** – the model degrades, the system notices, retrains itself and recovers.

Everything runs with `make`.

## Keep the drift part away from the team's data

Part 3 retrains and promotes a new champion. With the DagsHub lines in `.env`, `make` would do that in the
**shared** MLflow and every API would switch to the demo's storm model. So, in the terminal used for the demo:

```bash
make d-down                                   # stop your normal stack first (same ports)
export COMPOSE_PROJECT_NAME=skysat-demo       # separate Docker project: own database and local MLflow
alias dmake='make MLFLOW_TRACKING_URI= MLFLOW_TRACKING_USERNAME= MLFLOW_TRACKING_PASSWORD='
```

- `COMPOSE_PROJECT_NAME`: your normal prediction log and local MLflow are untouched, no `make d-wipe` needed.
- `dmake`: empty MLflow settings on the command line override `.env`, so everything uses the local MLflow.
  `.env` itself is not changed.
- Part 2 uses plain `make` on purpose: the Kaggle run belongs in the team's MLflow (experiment `skysat-kaggle`)
  and never touches the champion.

## Before recording (~1 min)

```bash
dmake d-up                                    # the stack (3 API copies, Nginx, PostgreSQL, Grafana, MLflow)
dmake ct-local ARGS="--model mlp"             # first champion in the local MLflow (~30 s)
dmake simulate                                # 2,000 normal passengers -> Grafana shows the "healthy" baseline
```

Also needed: `data/playground-series-s6e10.zip` (Kaggle → competition → Data; never committed) and a Kaggle
token. Open DagsHub (Models, Experiments) and Grafana at http://localhost:3000 (time range **Last 5 minutes**).

## The demo

| # | Command / screen | What the audience sees |
|---|---|---|
| 1 | DagsHub → Models → `skysat-satisfaction` | 5 model types from CT, their ROC AUC, the `champion` alias on the best (MLP) |
| 2 | `make submission` | competition track: 5-fold CV ROC AUC per fold → `artifacts/kaggle/submission.csv` (~75 s) |
| 3 | `make submit MSG="..."` then `make kaggle-scores` | the submission is accepted; its public leaderboard score |
| 4 | `dmake smoke` | one passenger → *satisfied*, probability, model version, which API copy answered |
| 5 | `dmake simulate MODE=drift` | winter storm (delays, worse ground service): accuracy drops in Grafana |
| 6 | `dmake retrain ARGS=--check-only` | PSI per feature: exactly the 9 features the storm changes are flagged |
| 7 | `dmake retrain` | retrain, both models scored on the same holdout, the better one becomes champion |
| 8 | `dmake simulate MODE=drift SEED=2` | new storm passengers, new model: accuracy recovers, Grafana shows version 2 |

- The Kaggle model is **not** one of the 5 CT models: the competition files have other columns, so the competition
  track ([training/competition.py](../training/competition.py)) trains its own gradient boosting model. Same idea
  (choose by validation score), separate pipeline.
- The retrained storm model from part 3 is not sent to Kaggle either: it is fitted to simulated storm traffic and
  lives only in the demo's local MLflow.
- `SEED=2` sends passengers the new model has not been trained on (the same seed would flatter it).
- Every `make submit` uses one of the 10 daily Kaggle submissions.

## Afterwards

```bash
make d-wipe                                   # answer y: deletes ONLY the skysat-demo project (COMPOSE_PROJECT_NAME)
unset COMPOSE_PROJECT_NAME; unalias dmake     # or just close this terminal
```

## Measured (rehearsal 2026-10-22, MacBook, Docker Desktop)

| Phase | Serving version | Accuracy (real answers) |
|---|---|---|
| Normal traffic | 1 | 96.25% |
| Storm, before retraining | 1 | 91.75% |
| Storm, after retraining (`SEED=2`) | 2 | 94.75% |

- Drift check: PSI 0.44–1.32 on the 9 storm features, ≤ 0.004 on the other 13.
- Quality gate on the same recent holdout: champion ROC AUC 0.9557, challenger 0.9825 → promoted.
- Kaggle: CV ROC AUC 0.95853 ± 0.00053 (same file as the team's earlier submission, sha256 d95f7bcaa8c8;
  public score 0.95789).
- Timings: preparation ~70 s, `make submission` ~75 s, each `simulate` ~18 s, `retrain` ~28 s, the APIs switch
  to the new champion in ≤ 10 s. Identical numbers in two runs. Waiting moments are cut from the video.
- The shared DagsHub registry was checked afterwards: unchanged.

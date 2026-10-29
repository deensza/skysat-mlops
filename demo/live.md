# SkySat – Live demo (only if the video is not accepted)

The assignment asks for "a RUNNING code example". Plan A is the recorded video ([video.md](video.md)). Use this
file if the instructor wants to see it run live. Same scenes, same commands ([README.md](README.md) runbook),
but everything that can fail or wait is done **before** we go on stage.

Open points:

- [ ] Rehearse this file once on the laptop we present with, on the classroom (or phone hotspot) network
- [ ] Keep the video file on the laptop as a fallback (not only online)

## What can go wrong live, and the answer

| Risk | Answer |
|---|---|
| No internet / DagsHub slow | Scene 1 (DagsHub) from screenshots in [assets/](assets/). Scenes 2–7 run fully offline: local MLflow, `dmake` |
| Docker images not pulled / built | `dmake d-up` the evening before and once more 15 min before the talk: nothing is downloaded on stage |
| Something hangs | Switch to the video at that scene; say what would happen |
| Laptop slow under 3 API copies | Close everything else; measure the stack's RAM in the rehearsal (`docker stats`) |
| A secret appears on screen | Never `cat .env`; terminal history cleared; DagsHub already logged in, no token typing |
| Waiting moments | `retrain` ~30 s, `simulate` ~20 s: talk over them (lines below) |

## 15 minutes before

```bash
make d-down                                   # your normal stack off (same ports)
export COMPOSE_PROJECT_NAME=skysat-demo
alias dmake='make MLFLOW_TRACKING_URI= MLFLOW_TRACKING_USERNAME= MLFLOW_TRACKING_PASSWORD='
make d-wipe                                   # y: clean demo project (deletes ONLY skysat-demo)
dmake d-up && dmake ct-local ARGS="--model mlp" && dmake simulate
```

Open in tabs, in this order: DagsHub Models, Grafana (http://localhost:3000, Last 5 minutes), local MLflow
(http://localhost:5050), and the terminal with a large font.

## On stage (~5 min)

Same scenes as [video.md](video.md), live. Talk over the waiting:

- While `simulate MODE=drift` runs: "the simulator replays passengers the model has never seen, after a winter
  storm: delays get longer, ground-service ratings drop."
- While `retrain` runs: "it reads the last 2,000 predictions from PostgreSQL, measures drift per feature, retrains
  on the old data plus the recent passengers, and scores both models on the same holdout."
- While waiting for Grafana to refresh: "each API copy checks MLflow every 10 seconds; no restart, no deploy."

## Afterwards

```bash
make d-wipe                                   # y: deletes ONLY the skysat-demo project
unset COMPOSE_PROJECT_NAME; unalias dmake
```

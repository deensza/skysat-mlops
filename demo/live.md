# SkySat – Live demo (only if the video is not accepted)

The assignment asks for "a RUNNING code example". Plan A is the recorded video ([video.md](video.md)). Use this
file if the instructor wants to see it run live. Same scenes, same commands ([README.md](README.md) runbook), but
everything that can fail or wait is done **before** we go on stage.

Open points:

- [ ] Rehearse this file once on the laptop we present with, on the classroom (or phone hotspot) network
- [ ] Keep the video file on the laptop as a fallback (not only online)

## What can go wrong live, and the answer

| Risk | Answer |
|---|---|
| No internet / Hugging Face blocked | The base model is already in the `llm-models` volume (or `HF_HUB_OFFLINE=1` with a local copy); everything runs offline against the local MLflow |
| No NVIDIA GPU on the presenting laptop | Do not train live: v1 and v2 are already in the demo MLflow; show the training output from the video |
| Docker images not pulled / model not downloaded | `dmake d-up` the evening before and 15 min before: nothing is downloaded on stage |
| A review takes seconds on CPU | `--per-month` 30 for the live replay; talk over the wait |
| Something hangs | Switch to the video at that scene; say what would happen |
| A secret appears on screen | Never `cat .env`; terminal history cleared |

## 15 minutes before

```bash
make d-down
export COMPOSE_PROJECT_NAME=skysat-demo
alias dmake='make MLFLOW_TRACKING_URI= MLFLOW_TRACKING_USERNAME= MLFLOW_TRACKING_PASSWORD='
dmake d-up && dmake smoke                     # answers once v1 is the champion in the demo MLflow
dmake llm-replay MONTHS=1 PER_MONTH=30        # healthy baseline in Grafana
```

Open in tabs: MLflow (http://localhost:5050), Grafana (http://localhost:3000, Last 1 hour), the terminal with a large font.

## On stage (~5 min)

Same scenes as [video.md](video.md). Talk over the waiting:

- While `llm-replay SCENARIO=harsh` runs: "the replay sends real reviews from 2019, rewritten as a harsher survey would
  rate them: lower scores, fewer recommendations. The model does not know the world changed."
- While `llm-drift` runs: "it reads aggregates from the server, no review text, and compares them with the gate."
- While v2 is shown: "only a 17 MB adapter changed; the API loads it next to the old one, no restart."

## Afterwards

```bash
make d-wipe                                   # y: deletes ONLY the skysat-demo project
unset COMPOSE_PROJECT_NAME; unalias dmake
```

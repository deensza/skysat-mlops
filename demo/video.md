# SkySat – Demo video

The assignment asks for "a RUNNING code example". We show it as a **~3-minute video** (waiting moments cut),
played after the slides and narrated live. Commands and safety setup: [README.md](README.md) (the runbook).
This file: what to film, what to say, how to record, where the video goes.

Open points (tick when done):

- [ ] Rehearsal 1: run every scene once, fill in "Rehearsal log" below
- [ ] Update [README.md](README.md) table row 1: it still says 5 models and an MLP champion (now: 8 candidates,
      champion v8 tuned gradient boosting)
- [ ] Record, cut, upload; put the link in [docs/presentation.md](../docs/presentation.md) slide 11 and the main README
- [ ] Screenshots for the slides into [assets/](assets/)

## Scenes

| # | Time | Screen | Command | Say |
|---|---|---|---|---|
| 1 | 0:00–0:40 | DagsHub → Experiments, then Models → `skysat-satisfaction` | – | "CT trained 8 candidates on GitHub's machines. Each run: settings, scores, the model file. The quality gate gave the `champion` alias to the best: version 8." |
| 2 | 0:40–1:10 | Terminal, then browser | `dmake smoke` | "One passenger through Nginx: satisfied, the probability, which model version, which of the 3 API copies answered." |
| 3 | 1:10–1:30 | Grafana (Last 5 minutes) | `dmake simulate` (already run before recording) | "2,000 normal passengers. Accuracy with their real answers: 96%. This is the healthy baseline." |
| 4 | 1:30–2:00 | Terminal → Grafana | `dmake simulate MODE=drift` | "A winter storm: longer delays, worse ground service. Accuracy drops to about 92%. The model does not know the world changed." |
| 5 | 2:00–2:20 | Terminal | `dmake retrain ARGS=--check-only` | "The drift check compares every feature with the training data. Exactly the 9 features the storm changes are flagged." |
| 6 | 2:20–2:45 | Terminal → DagsHub/local MLflow | `dmake retrain` | "Retrain on recent passengers. Old and new model are scored on the same holdout; the new one is better, so it becomes champion." |
| 7 | 2:45–3:05 | Grafana | `dmake simulate MODE=drift SEED=2` | "New storm passengers the model has never seen. Version 2 is serving, accuracy recovers to about 95%. No restart, no code change." |
| (8) | optional | Terminal | `make submission` | Kaggle track; only if time allows: it adds ~75 s of waiting. |

- Numbers in "Say" come from the 2026-10-22 rehearsal; replace them with the rehearsal log below.
- Scene 1 uses the **shared** DagsHub (read-only). Scenes 2–7 use `dmake` = the demo's own Docker project and
  local MLflow, so the team's champion is never touched (see [README.md](README.md)).

## Recording checklist

- **No secrets on screen:** never open `.env`, `~/.kaggle/`, DagsHub tokens or `docker compose config` output.
  Check every frame before uploading.
- **No raw data on screen:** the competition rules forbid redistributing the data. Aggregates, charts and single
  example passengers are fine; scrolling through CSV files is not.
- Screen 1920×1080, terminal font ≥ 16 pt, browser zoom ~125%, notifications off, one clean desktop.
- Record each scene separately (easier to redo one), cut the waiting, join them.
- Tools: OBS Studio (Linux/macOS/Windows, free) or the system recorder; Kdenlive/Shotcut or iMovie to cut.

## Where the video goes

**Not into git.** A video is 50–500 MB; GitHub refuses files over 100 MB and every clone downloads every version
forever. Upload it to YouTube (unlisted) or Google Drive (anyone with the link) and put the **link** in the
slides, the main README and Moodle. Small screenshots for the slides go in [assets/](assets/) (git ignores video
files there).

## Rehearsal log

| Date | Who / machine | Scene | Result (numbers, time) | Problems |
|---|---|---|---|---|
| 2026-10-22 | MacBook, Docker Desktop | all | normal 96.25%, storm 91.75%, after retrain 94.75%; PSI 0.44–1.32 on 9 features | – |
| | | | | |

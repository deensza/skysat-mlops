# SkySat – Demo video

The assignment asks for "a RUNNING code example". We show it as a **~3-minute video** (waiting moments cut), played after
the slides and narrated live. Commands and safety setup: [README.md](README.md). This file: what to film, what to say.

Open points:

- [ ] Rehearse every scene once on the demo machine; fill in the rehearsal log below
- [ ] Record, cut, upload; put the link in [docs/presentation.md](../docs/presentation.md) slide 10 and the main README
- [ ] Screenshots for the slides into [assets/](assets/)

## Scenes

| # | Time | Screen | Command | Say |
|---|---|---|---|---|
| 1 | 0:00-0:30 | MLflow → Models → `skysat-reviews` | – | "A 1.5B LLM with a 17 MB LoRA adapter, trained in 5 minutes. The gate, in the local demo run shown here: MAE 0.88, recommended 92.0%, complaint 71.5%. Version 1 is the champion." |
| 2 | 0:30-0:50 | Terminal | `dmake smoke` | "One review through Nginx: score, recommended, main complaint, model version, latency." |
| 3 | 0:50-1:10 | Grafana | `dmake simulate FROM=2019-01 MONTHS=1 PER_MONTH=30` | "Real 2019 reviews with their true labels. Real drift does not hurt this model: MAE 0.95, recommended 95%." |
| 4 | 1:10-1:40 | Terminal → Grafana | `dmake simulate FROM=2019-02 MONTHS=1 PER_MONTH=300 SCENARIO=harsh` | "A harsher survey. Live MAE and recommended accuracy cross the red lines: MAE 1.15, recommended 89%." |
| 5 | 1:40-2:00 | Terminal | `dmake llm-drift` | "The drift check reads aggregates, never text, and says DRIFT." |
| 6 | 2:00-2:30 | Terminal → MLflow | `dmake queue-stats` (`ready` >= 300), then `make llm-train LLM_MLFLOW=http://localhost:5050 ARGS="--replay 0 --epochs 3"` | "Retrain on the new reviews only. On held-back reviews: MAE 0.60, recommended 93%; the champion on the same reviews: 1.13 and 90%. It beats the champion, so it is promoted." |
| 7 | 2:30-3:00 | Grafana | `dmake simulate FROM=2019-03 MONTHS=1 PER_MONTH=30 SCENARIO=harsh` | "The API loads the new adapter by itself. Version 2 is serving and accuracy recovers: on the server, MAE 0.55 and recommended 97% on 165 new reviews." |

- Numbers in "Say" are the ones measured through the server on 16 and 17 Nov 2026 ([docs/llm.md](../docs/llm.md); on
  the server the retrained adapter was version 4, in the demo it is version 2). The demo machine will give other
  numbers: replace them with the rehearsal log below.
- Scene 4 sends 300 reviews because the trainer needs 300 new ones (`--min-new`), and a review can be claimed only
  after the model has scored it. At about 20 s a review that is about 1 h 40 min on a laptop CPU: send them long
  before filming scene 6 and cut the wait.
- Scene 6 needs both additions. `LLM_MLFLOW=http://localhost:5050` trains into the demo stack's MLflow, the one the
  demo API reads; plain `make llm-train` writes to a sqlite file it does not read. `--replay 0 --epochs 3` trains on
  the new reviews only; the default (500 replayed reviews, one epoch) failed the gate on drifted reviews on 16 Nov.
- The rented GPU is not in the video's live part: show the Drift check run of 17 Nov 2026 on GitHub that rented one
  as a still ([README.md](README.md), step 9).
- Use `dmake` and the local MLflow so the team's champion is never touched.

## Recording checklist

- **No secrets on screen:** never open `.env`, tokens or `docker compose config` output. Check every frame.
- **No raw data on screen:** the reviews are scraped text under DbCL-1.0. Aggregates, charts and one or two example
  reviews are fine; scrolling through the CSV is not.
- Screen 1920×1080, terminal font >= 16 pt, browser zoom ~125%, notifications off.
- Record each scene separately, cut the waiting, join them. Tools: OBS Studio, Kdenlive/Shotcut.

## Where the video goes

**Not into git.** Upload it to YouTube (unlisted) or Google Drive and put the **link** in the slides, the main README and
Moodle. Small screenshots go in [assets/](assets/) (git ignores video files there).

## Rehearsal log

| Date | Who / machine | Scene | Result (numbers, time) | Problems |
|---|---|---|---|---|
| | | | | |

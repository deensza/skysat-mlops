# SkySat – Demo video

The assignment asks for "a RUNNING code example". We show it as a **~3-minute video** (waiting moments cut), played after
the slides and narrated live. Commands and safety setup: [README.md](README.md). This file: what to film, what to say.

Open points:

- [ ] Rehearse every scene once on the demo machine; fill in the rehearsal log below
- [x] Record, cut, upload; put the link in [docs/presentation.md](../docs/presentation.md) slide 10 and the main README
      (17 Nov 2026: two videos, see "Recorded videos" below)
- [ ] Screenshots for the slides into [assets/](assets/)

## Scenes

| # | Time | Screen | Command | Say |
|---|---|---|---|---|
| 1 | 0:00-0:30 | MLflow → Models → `skysat-reviews` | – | "A 1.5B LLM with a 17 MB LoRA adapter, trained in 5 minutes. The gate: MAE 0.91, recommended 94.5%, complaint 68%. Version 1 is the champion." |
| 2 | 0:30-0:50 | Terminal | `dmake smoke` | "One review through Nginx: score, recommended, main complaint, model version, latency." |
| 3 | 0:50-1:10 | Grafana | `dmake simulate FROM=2019-01 MONTHS=1 PER_MONTH=30` | "Real 2019 reviews with their true labels. Real drift does not hurt this model: MAE stays under 1." |
| 4 | 1:10-1:40 | Terminal → Grafana | `dmake simulate FROM=2019-01 MONTHS=1 PER_MONTH=150 SCENARIO=harsh` | "A harsher survey. Live MAE and recommended accuracy cross the red lines: MAE 1.18, recommended 89.5%." |
| 5 | 1:40-2:00 | Terminal | `dmake llm-drift` | "The drift check reads aggregates, never text, and says DRIFT." |
| 6 | 2:00-2:30 | Terminal → MLflow | `make llm-train` | "Retrain on harsh reviews. Version 2: MAE 0.41, recommended 95.5%. It beats the champion, so it is promoted." |
| 7 | 2:30-3:00 | Grafana | `dmake simulate FROM=2019-04 MONTHS=1 PER_MONTH=30 SCENARIO=harsh` | "The API loads the new adapter by itself. Version 2 is serving and accuracy recovers." |

- Numbers in "Say" are the measured ones in [docs/llm.md](../docs/llm.md); replace them with the rehearsal log below.
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

## Recorded videos (17 Nov 2026)

Both are silent (narrated live), with waiting moments cut or sped up. Folder: [Google Drive](https://drive.google.com/drive/folders/1kfRWCZTZPatf758JnG09AcfO2DWUIb80).

**[The server and its retraining loop](https://drive.google.com/file/d/1Ulbq2-DxDoxo9LLP3ISe-qSe07x_d64b/view)** (2:18), recorded from a laptop against the team's server, no GPU needed:

| Time | Screen | What it shows |
|---|---|---|
| 0:00 | GitHub issue 18 | the drift check opened "Drift detected: the serving model fell below the gate" by itself |
| 0:08 | Actions → Drift check run 44111118456 | the drift table (DRIFT on version 4), then the job `retrain / train`: a rented RTX 3080 Ti, "serving model 4 -> 6", "Trainer pods left behind: 0" |
| 0:35 | DagsHub → Models | both registered models (`skysat-satisfaction`, the tabular model; `skysat-reviews`, the LLM), versions 1-6, champion 6, its metrics and the `adapter` files |
| 1:07 | Swagger → `GET /model` | the server serves version 6 on Qwen/Qwen2.5-1.5B-Instruct |
| 1:22 | Swagger → `POST /score` | one made-up review; the wait is sped up 25×; answer: score 1, recommended no, complaint ground service, version 6, about 37 s |
| 1:57 | Grafana | requests, live MAE and recommended accuracy per hour, model version over time |

The server's address is blurred in every frame (it is not in this public repository on purpose) and the gateway's login
dialog is cut out.

**[The Kaggle competition, the project's first model](https://drive.google.com/file/d/1dPYeIG6mb0o7lfYjoIIgoqsWV4Z0CbQd/view)** (1:46): Playground S6E10 with the tabular model the project
started with, before the LLM: the team's submissions (public 0.95789 → 0.95982 → 0.96091), the run in MLflow
(5-fold CV 0.96113) and the leaderboard (rank 1341 of 3,705 at recording time). Final private score: 0.96151.

## Rehearsal log

| Date | Who / machine | Scene | Result (numbers, time) | Problems |
|---|---|---|---|---|
| | | | | |

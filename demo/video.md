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
| 1 | 0:00-0:30 | MLflow → Models → `skysat-reviews` | – | "A 1.5B LLM with a 17 MB LoRA adapter, trained in 5 minutes. The gate: MAE 0.91, recommended 94.5%, complaint 68%. Version 1 is the champion." |
| 2 | 0:30-0:50 | Terminal | `dmake smoke` | "One review through Nginx: score, recommended, main complaint, model version, latency." |
| 3 | 0:50-1:10 | Grafana | `dmake llm-replay` | "Real 2019 reviews with their true labels. Real drift does not hurt this model: MAE stays under 1." |
| 4 | 1:10-1:40 | Terminal → Grafana | `dmake llm-replay SCENARIO=harsh` | "A harsher survey. Live MAE and recommended accuracy cross the red lines: MAE 1.18, recommended 89.5%." |
| 5 | 1:40-2:00 | Terminal | `dmake llm-drift` | "The drift check reads aggregates, never text, and says DRIFT." |
| 6 | 2:00-2:30 | Terminal → MLflow | `make llm-train LLM_SPLIT=live-2019 SCENARIO=harsh ...` | "Retrain on harsh reviews. Version 2: MAE 0.41, recommended 95.5%. It beats the champion, so it is promoted." |
| 7 | 2:30-3:00 | Grafana | `dmake llm-replay SCENARIO=harsh FROM=2019-04` | "The API loads the new adapter by itself. Version 2 is serving and accuracy recovers." |

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

## Rehearsal log

| Date | Who / machine | Scene | Result (numbers, time) | Problems |
|---|---|---|---|---|
| | | | | |

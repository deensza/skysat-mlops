# SkySat – Kaggle track video

A short (~2 min) video of the competition pipeline: Kaggle Playground S6E10 *Predicting Airline Satisfaction*,
metric ROC AUC, deadline **2026-10-31 23:59 UTC**. It shows that the competition model is trained, tracked and
submitted with the same MLOps habits as the serving model. Code: [training/competition.py](../training/competition.py).

Open points (tick when done):

- [ ] Final submission made (`make submit`) and its public score known
- [ ] Rehearsal: run every scene once, fill in "Rehearsal log"
- [ ] Record, cut, upload; put the link on slide 13 of [docs/presentation.md](../docs/presentation.md)

## Rules for what may be on screen (competition rules)

- **No competition rows.** Never open `data/playground-series-s6e10.zip`, `train.csv`, `test.csv` or
  `artifacts/kaggle/submission.csv` on screen: the data may only go to competition participants. Scores, column
  *names*, row counts and plots of aggregates are fine.
- **No secrets**: no `~/.kaggle/`, no `.env`, no DagsHub token in the terminal history you scroll past.
- The video file stays out of git (`.gitignore` has `*.mp4 *.mov *.mkv *.webm`).

## Scenes

| # | Time | Screen | Command | Say |
|---|---|---|---|---|
| 1 | 0:00–0:20 | Kaggle competition page (Overview tab) | – | "Same problem, different data: Kaggle's synthetic passengers, other columns, scored by ROC AUC on hidden labels." |
| 2 | 0:20–0:50 | Terminal | `make submission` (cut the waiting) | "5-fold cross-validation on the competition's training rows: each fold model is scored on rows it never saw, then predicts the test set. The submission averages the 5 predictions." |
| 3 | 0:50–1:10 | Terminal output | – | "CV ROC AUC per fold, the mean and the spread, and the submission's sha256, so every upload can be traced back to a run." |
| 4 | 1:10–1:30 | MLflow → experiment `skysat-kaggle` | – | "Every attempt is an MLflow run: settings, CV scores, git commit, submission hash. Here is what we tried and what was worse, for example adding the original dataset." |
| 5 | 1:30–1:50 | Terminal | `make submit MSG="..."`, then `make kaggle-scores` | "Upload with a message that says what changed; the public score comes back next to it." |
| 6 | 1:50–2:05 | Kaggle → Leaderboard / My submissions | – | "Public score vs. our CV: close, so the CV is honest. The final ranking uses the private part of the test set." |

- Numbers to say: fill in from the rehearsal; before: hgb CV 0.95853, public 0.95789; now: hgb + LightGBM, CV 0.95905 ± 0.00056 (sha256 39ec2ef675bf).
- Waiting in scene 2 is cut; say how long it really took.

## Rehearsal log

| Date | Scene | What happened | Fix |
|---|---|---|---|
| | | | |

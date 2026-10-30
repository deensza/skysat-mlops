# SkySat – Kaggle track video

A short (~2 min) video of the competition pipeline: Kaggle Playground S6E10 *Predicting Airline Satisfaction*,
metric ROC AUC, deadline **2026-10-31 23:59 UTC**. It shows that the competition model is trained, tracked and
submitted with the same MLOps habits as the serving model. Code: [training/competition.py](../training/competition.py).

Open points (tick when done):

- [x] Final submission made (`make submit`, 2026-10-30): public 0.96091, CV 0.96113, sha256 4a1ee109a42a, DagsHub run 8f95fdc70c8a
  (earlier the same day: public 0.95982, CV 0.96013, sha256 ad609290c4a3, run 27ebe4bbd893)
- [ ] Rehearsal: run every scene once, fill in "Rehearsal log"
- [x] Record, cut, upload: [Google Drive](https://drive.google.com/drive/folders/1WBdX3AQdVqQV6cO-wTW8shrivhatKatB). Link on slide 13 of [docs/presentation.md](../docs/presentation.md)

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

- Numbers to say: fill in from the rehearsal; before: hgb CV 0.95853, public 0.95789; then: hgb + LightGBM + original-data rates, CV 0.96013 ± 0.00053 (sha256 ad609290c4a3), public 0.95982; now: LightGBM + XGBoost + RealMLP with original-data lookups and in-fold target encoding, CV 0.96113 ± 0.00054 (sha256 4a1ee109a42a), public 0.96091.
- Waiting in scene 2 is cut; say how long it really took.

## Rehearsal log

| Date | Scene | What happened | Fix |
|---|---|---|---|
| | | | |

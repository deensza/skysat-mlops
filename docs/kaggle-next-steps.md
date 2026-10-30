# Kaggle: fastest ways to a better score (note for Deniz)

State on 2026-10-30: **public 0.96091, CV 0.96113 ± 0.00054**. The model is LightGBM + XGBoost + RealMLP, with their
logits averaged (RealMLP has the weight of the two trees together). DagsHub run `8f95fdc70c8a`, submission sha256
`4a1ee109a42a`. The public #1 is 0.96451. Deadline **2026-10-31 23:59 UTC**. All code is in
[training/competition.py](../training/competition.py), and its docstring lists every result.

## What the top of the leaderboard does that we don't

From reading the top public notebooks (read only, none of them was run):

1. **Many different models, combined by a second model (stacking).** They train 10–30 models (CatBoost, several
   GBDT settings, RealMLP, TabM and other neural networks). Then a simple model (logistic regression or ridge) learns
   from the out-of-fold predictions how to combine them. We use 3 models with fixed weights.
2. **Several seeds per model**, averaged.
3. **Public out-of-fold predictions** from other people's notebooks, added as stacking inputs. The rules allow it
   because those files are public. We chose not to use them, so that our result comes only from our own pipeline.

Our measurements say the gain comes from **models that make different mistakes**, not from more of the same kind.
LightGBM and XGBoost logits correlate at 0.999, so XGBoost in place of HGB gained only +0.00011. RealMLP correlates at
0.993–0.995 with the trees and added +0.00057.

## Do these in this order (biggest gain per hour first)

| # | What | Expected gain | Cost |
|---|---|---|---|
| 1 | **CatBoost** as a 4th model (`catboost`, Apache-2.0: add it to `requirements-train.txt`) | +0.0001–0.0003 | ~1 h |
| 2 | **Stacker**: logistic regression on the models' out-of-fold logits, in place of fixed weights | +0.0001–0.0002 | ~1 h, no retraining |
| 3 | **RealMLP with 2–3 seeds** (change `random_state`, average the seeds) | ~+0.0001 | ~1.5 h of runs |
| 4 | **TabM** (a second neural network; pytabkit has `TabM_D_Classifier`) | +0.0002–0.0004, uncertain | half a day |

Steps 1–3 together should give about **0.9615–0.962 public**. Reaching 0.965 is not realistic: it is above the
current #1.

How to add a model (CatBoost as the example):
- Add `"catboost"` to `MODELS` and a branch in `fit_predict`. Copy the XGBoost branch: same inputs, early stopping
  on the last 10%, categoricals passed as `cat_features`.
- For the stacker: `cross_validate` already keeps every model's out-of-fold predictions. Fit
  `LogisticRegression` on their logits **inside each fold**: fit on the training rows' OOF predictions, score on
  the held-out ones. Never fit it on all OOF rows and then report that score, which leaks.

## Already tried and worse (don't repeat)

- Adding the original dataset's rows to training.
- Satisfied-share rates for pairs of columns.
- More target-encoded combinations than the current `TE_COMBOS`.
- A model trained on the original data as an extra input: +0.00009, within noise.
- A second RealMLP on other inputs: 0.96117 vs 0.96125 for one RealMLP with a better weight (fold 1).

## How to work

- **Screen on fold 1 first** (about 5 min), with a scratch script outside the repo that saves each model's fold-1
  predictions (e.g. to parquet), so trying blends needs no retraining. Only run the full 5 folds for an idea that wins
  on fold 1.
- Quick full runs without logging: `make submission ARGS="--no-mlflow"`. The final run is plain
  `make submission` (logs to DagsHub), then `make submit MSG="what changed, CV x.xxxxx"`.
- Keep a change only if the CV is better **in most of the 5 folds**, not just on the mean. Under ~0.0002, public
  leaderboard differences are mostly noise, and the private part of the test set decides the ranking.
- Pick the 2 final submissions on Kaggle (Submissions tab) **by CV**, not by public score.
- `OMP_WAIT_POLICY=PASSIVE` avoids an OpenMP slowdown when LightGBM and PyTorch run in the same process.
- LightGBM rejects feature names with commas or other JSON characters.

## Rules (competition)

- Competition and original-dataset rows never leave our PCs: not in git, DagsHub/MLflow, or chats. Log only
  scores, counts and hashes.
- Any new library must be open source and allow commercial use (CatBoost and pytabkit: Apache-2.0).
- External data or public notebooks' outputs must be public and free for all participants (rule 2.6).

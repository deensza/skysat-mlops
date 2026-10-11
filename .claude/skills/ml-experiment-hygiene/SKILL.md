---
name: ml-experiment-hygiene
description: Checklist for training, evaluating, and registering the SkySat satisfaction model without fooling ourselves — data leakage, reproducibility, honest metrics, MLflow logging, model registry. Use when writing or reviewing anything in training/, or when reporting model metrics.
---

# ML experiment hygiene

A model number is a claim. Make it reproducible and measured, not asserted.

## Data

- Split **before** any fitting step (imputing, scaling, encoding, feature selection). Fit preprocessing on train only, apply to validation/test. Prefer a single sklearn `Pipeline` so this cannot be done wrong.
- Fixed `random_state` everywhere (split, model, any sampling). Stratify the split on the target.
- Keep the held-out test set untouched until the final evaluation; tune on validation / cross-validation.
- Check the columns for target leakage (anything only known after the passenger answered the survey) and drop or flag it.
- Look at the data before modelling: class balance, missing values, dtypes. Print them; don't assume.
- Never commit dataset files. `data/` is gitignored on purpose.

## Metrics

- Report more than accuracy: at least precision, recall, F1, ROC-AUC, plus the class balance and a naive baseline (majority class) so the numbers have context.
- Metrics come from running the evaluation code, with the raw output pasted — never from memory or estimation.
- A score that is surprisingly high is a leakage suspect first. Re-check the split and features before celebrating.

## MLflow

- One run per training attempt. Log: all hyperparameters, data version/row counts and split seed, every metric above, and the fitted pipeline as the model artifact (with an input example and signature).
- Tag runs with the git commit and the step/purpose (`training`, `retraining`).
- Register a model only when it clears a stated threshold, and compare against the current registered version before promoting. Use registry aliases (e.g. `champion`) rather than hard-coding version numbers in the API.
- The API loads the model from the registry by alias, so retraining + promotion changes production behaviour without a code change.

## Retraining and drift

- Retraining must use the same pipeline code and metrics as initial training — one code path, parameterised by data.
- Drift checks compare a reference window (training data) with a recent window (logged predictions), per feature, with the statistic and threshold written down. The simulator's drift mode is the test input; confirm drift is detected on drifted traffic **and not** detected on normal traffic.

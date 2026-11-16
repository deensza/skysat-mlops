---
name: ml-experiment-hygiene
description: Checklist for training, evaluating, and registering the SkySat airline-review LoRA model without fooling ourselves — data leakage, reproducibility, honest metrics, MLflow logging, model registry. Use when writing or reviewing anything in llm/train_lora.py, llm/trainer.py, llm/gate.py or llm/reviews.py, or when reporting model metrics.
---

# ML experiment hygiene

A model number is a claim. Make it reproducible and measured, not asserted.

## Data

- Split **before** any fitting step. Fit on train only, apply to validation/test (`llm/reviews.py` splits by date and airline).
- Fixed seed everywhere (split, training, any sampling).
- Keep the held-out test set untouched until the final evaluation; tune on validation / cross-validation.
- Check the inputs for target leakage and drop or flag it.
- Look at the data before modelling: class balance, missing values. Print them; don't assume.
- Never commit dataset files. `data/` is gitignored on purpose.

## Metrics

- Report more than one number (score MAE, recommended accuracy, complaint accuracy, as in `llm/gate.py`), plus the class balance and a naive baseline so the numbers have context.
- Metrics come from running the evaluation code, with the raw output pasted — never from memory or estimation.
- A score that is surprisingly high is a leakage suspect first. Re-check the split and features before celebrating.

## MLflow

- One run per training attempt. Log: all hyperparameters, data version/row counts and split seed, every metric above, and the LoRA adapter as the model artifact.
- Tag runs with the git commit and the step/purpose (`training`, `retraining`).
- Register a model only when it clears a stated threshold, and compare against the current registered version before promoting. Use registry aliases (e.g. `champion`) rather than hard-coding version numbers in the API.
- The API loads the model from the registry by alias, so retraining + promotion changes production behaviour without a code change.

## Retraining and drift

- Retraining must use the same training code and metrics as initial training — one code path, parameterised by data.
- Drift checks compare a reference window (training data) with a recent window (logged predictions), with the statistic and threshold written down. The simulator's harsh scenario is the test input; confirm drift is detected on drifted traffic **and not** detected on normal traffic.

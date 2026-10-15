"""Where does the PyTorch MLP stand? Compare it with simple models on the same split.

Trains on data/processed/train.csv and scores on validation.csv (the same files and the same
preprocessing as training.train), with several seeds so the run-to-run noise is visible:

    python -m training.compare            # or: make compare
    python -m training.compare --seeds 5 --epochs 30

Models: majority class (the "no skill" line), logistic regression, gradient boosting
(scikit-learn's HistGradientBoostingClassifier) and the MLP from training.train.
Nothing is saved or registered; this only prints a table to guide decisions such as the
MLflow quality-gate threshold.
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from skysat import schema
from skysat.data import PROCESSED_DIR
from training.train import DEFAULT_PARAMS, _standardize, run_training, to_matrix


def sklearn_models(seed: int) -> dict:
    return {
        "majority class": DummyClassifier(strategy="prior"),
        "logistic regression": LogisticRegression(max_iter=1000, random_state=seed),
        "gradient boosting": HistGradientBoostingClassifier(random_state=seed),
    }


def compare(train_df: pd.DataFrame, val_df: pd.DataFrame, seeds: list[int], params: dict | None = None) -> pd.DataFrame:
    """Validation ROC AUC for every model and seed. Returns one row per (model, seed)."""
    Xt, Xv = to_matrix(train_df), to_matrix(val_df)
    mean = Xt[:, _n_cat():].mean(0)  # scaling statistics from the training rows only
    std = Xt[:, _n_cat():].std(0) + 1e-6
    Xt, Xv = _standardize(Xt, mean, std), _standardize(Xv, mean, std)
    yt, yv = train_df[schema.TARGET].to_numpy(), val_df[schema.TARGET].to_numpy()

    rows = []
    for seed in seeds:
        for name, model in sklearn_models(seed).items():
            model.fit(Xt, yt)
            rows.append((name, seed, roc_auc_score(yv, model.predict_proba(Xv)[:, 1])))
        with tempfile.TemporaryDirectory() as tmp:  # the MLP also writes a model.pt; we only want its score
            _, metrics = run_training(train_df, val_df, {**(params or {}), "seed": seed}, Path(tmp))
        rows.append(("PyTorch MLP", seed, metrics["roc_auc"]))
    return pd.DataFrame(rows, columns=["model", "seed", "roc_auc"])


def _n_cat() -> int:
    return sum(len(v) for v in schema.CATEGORICAL_VALUES.values())


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    table = results.groupby("model", sort=False)["roc_auc"].agg(["mean", "std", "min", "max"])
    return table.round(4)


def main():
    ap = argparse.ArgumentParser(description="Compare the MLP with simple baselines")
    ap.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=DEFAULT_PARAMS["epochs"])
    args = ap.parse_args()

    paths = [args.data_dir / "train.csv", args.data_dir / "validation.csv"]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"ERROR: {p} not found. Run `make data` first.")
    train_df, val_df = pd.read_csv(paths[0]), pd.read_csv(paths[1])

    print(f"train rows {len(train_df):,} | validation rows {len(val_df):,} | seeds {args.seeds}")
    results = compare(train_df, val_df, list(range(args.seeds)), {"epochs": args.epochs})
    print("\nvalidation ROC AUC (mean / std / min / max over seeds):")
    print(summarize(results).to_string())


if __name__ == "__main__":
    main()

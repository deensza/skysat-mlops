"""Kaggle competition track: Playground Series S6E10, "Predicting Airline Satisfaction" (metric: ROC AUC).

Kept apart from the serving pipeline on purpose: the competition files have other columns (no "Inflight service",
a True/False target) and must stay on the team's PCs (competition rules: only participants may get the data),
so this runs locally and never on GitHub. Its MLflow runs go to their own experiment, `skysat-kaggle`, and are
never registered: these models cannot serve the API's passengers.

    make submission                        # 5-fold CV score + artifacts/kaggle/submission.csv
    make submit MSG="what changed"         # upload it to Kaggle (needs your Kaggle token)
    make kaggle-scores                     # your team's submissions and their public leaderboard scores

How it works:
    1. read train.csv / test.csv straight from data/playground-series-s6e10.zip (no unzipping)
    2. 5-fold stratified cross-validation on the competition train rows: every fold model is scored on
       competition rows it never saw, then predicts the test set; the submission averages the 5 predictions
    3. optional (--original) external data, allowed by rule 6.a (public and free): the original dataset
       (data/train.csv + test.csv) in the competition's columns, added to the TRAINING folds only

Chosen by 5-fold CV ROC AUC on the competition train rows (2026-10-20), defaults in DEFAULT_PARAMS:
    learning_rate 0.1, max_iter 100, max_leaf_nodes 31 (sklearn defaults)   0.95767 ± 0.00063
        + original dataset                                                  0.95727 ± 0.00057  (worse in all 5 folds)
    learning_rate 0.05, max_iter 1000, max_leaf_nodes 31                    0.95814 ± 0.00065
    learning_rate 0.05, max_iter 1000, max_leaf_nodes 63                    0.95843 ± 0.00061
        + original dataset                                                  0.95804 ± 0.00056
    learning_rate 0.05, max_iter 1000, max_leaf_nodes 127                   0.95853 ± 0.00053  <- default
max_iter is an upper limit: early stopping (on 10% of each training fold) ends training sooner.
"""

import argparse
import hashlib
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from skysat.data import DATA_DIR, PROJECT_ROOT

COMPETITION = "playground-series-s6e10"
ZIP = DATA_DIR / f"{COMPETITION}.zip"
OUT_DIR = PROJECT_ROOT / "artifacts" / "kaggle"
TARGET = "satisfaction"
CATEGORICAL = ["Gender", "Customer Type", "Type of Travel", "Class"]
SEED = 42

DEFAULT_PARAMS = {"learning_rate": 0.05, "max_iter": 1000, "max_leaf_nodes": 127}


def load_competition() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ZIP.exists():
        raise SystemExit(f"ERROR: {ZIP} not found. Join the competition on Kaggle and download its data into data/.")
    with zipfile.ZipFile(ZIP) as z:
        train = pd.read_csv(z.open("train.csv"))
        test = pd.read_csv(z.open("test.csv"))
    train[TARGET] = train[TARGET].astype(int)
    return train, test


def load_original(columns: list[str]) -> pd.DataFrame:
    """The original dataset (step 2's data/train.csv + test.csv) in the competition's columns and target."""
    parts = [pd.read_csv(DATA_DIR / name) for name in ("train.csv", "test.csv")]
    original = pd.concat(parts, ignore_index=True)
    original[TARGET] = (original[TARGET] == "satisfied").astype(int)
    return original[columns]  # drops the index column, "id" and "Inflight service" (not in the competition)


def features(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    X = df.drop(columns=["id", TARGET], errors="ignore")
    for col in CATEGORICAL:
        X[col] = X[col].astype(categories[col])  # same category codes in every frame
    return X


def model(params: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=SEED, **params)


def cross_validate(train: pd.DataFrame, test: pd.DataFrame, extra: pd.DataFrame | None, params: dict,
                   folds: int = 5) -> tuple[list[float], np.ndarray]:
    """Per-fold ROC AUC on held-out competition rows, and the test predictions averaged over the folds."""
    frames = [train, test] + ([extra] if extra is not None else [])
    categories = {c: CategoricalDtype(sorted(pd.concat([f[c] for f in frames]).dropna().unique())) for c in CATEGORICAL}
    X, y, X_test = features(train, categories), train[TARGET], features(test, categories)
    X_extra = features(extra, categories) if extra is not None else None

    scores, test_pred = [], np.zeros(len(test))
    for k, (fit_idx, val_idx) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, y)):
        X_fit, y_fit = X.iloc[fit_idx], y.iloc[fit_idx]
        if X_extra is not None:  # external rows only ever train; they are never scored
            X_fit, y_fit = pd.concat([X_fit, X_extra]), pd.concat([y_fit, extra[TARGET]])
        fitted = model(params).fit(X_fit, y_fit)
        scores.append(roc_auc_score(y.iloc[val_idx], fitted.predict_proba(X.iloc[val_idx])[:, 1]))
        test_pred += fitted.predict_proba(X_test)[:, 1] / folds
        print(f"  fold {k + 1}/{folds}: roc_auc {scores[-1]:.5f} ({fitted.n_iter_} rounds)", flush=True)
    return scores, test_pred


def write_submission(test: pd.DataFrame, pred: np.ndarray) -> Path:
    with zipfile.ZipFile(ZIP) as z:
        expected_ids = pd.read_csv(z.open("sample_submission.csv"))["id"]
    sub = pd.DataFrame({"id": test["id"], TARGET: pred})
    if not sub["id"].equals(expected_ids) or sub[TARGET].isna().any() or not sub[TARGET].between(0, 1).all():
        raise SystemExit("ERROR: the submission does not match sample_submission.csv (ids, NaN or range)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "submission.csv"
    sub.to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser(description="Cross-validate on the competition data and write a submission")
    ap.add_argument("--original", action="store_true", help="add the original dataset to training (was worse)")
    ap.add_argument("--learning-rate", type=float, default=DEFAULT_PARAMS["learning_rate"])
    ap.add_argument("--max-iter", type=int, default=DEFAULT_PARAMS["max_iter"])
    ap.add_argument("--max-leaf-nodes", type=int, default=DEFAULT_PARAMS["max_leaf_nodes"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-mlflow", action="store_true", help="do not log the run (quick experiments)")
    args = ap.parse_args()

    train, test = load_competition()
    extra = load_original(list(train.columns)) if args.original else None
    params = {"learning_rate": args.learning_rate, "max_iter": args.max_iter, "max_leaf_nodes": args.max_leaf_nodes}
    print(f"competition train {len(train):,} rows ({train[TARGET].mean():.1%} satisfied), test {len(test):,} rows, "
          f"original added: {0 if extra is None else len(extra):,} rows | {params}")

    scores, pred = cross_validate(train, test, extra, params, args.folds)
    path = write_submission(test, pred)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    print(f"CV roc_auc {np.mean(scores):.5f} ± {np.std(scores):.5f} | wrote {path.relative_to(PROJECT_ROOT)} "
          f"(sha256 {digest})")

    if not args.no_mlflow:
        import mlflow
        from training.track import git_commit
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or "http://localhost:5050")
        mlflow.set_experiment("skysat-kaggle")
        with mlflow.start_run(tags={"purpose": "kaggle", "git_commit": git_commit(), "submission_sha256": digest}):
            mlflow.log_params({**params, "model": "hist_gradient_boosting", "folds": args.folds,
                               "original_rows": 0 if extra is None else len(extra), "train_rows": len(train)})
            mlflow.log_metrics({"cv_roc_auc_mean": float(np.mean(scores)), "cv_roc_auc_std": float(np.std(scores)),
                                **{f"cv_roc_auc_fold{k + 1}": s for k, s in enumerate(scores)}})
        print(f"Logged to MLflow ({mlflow.get_tracking_uri()}, experiment skysat-kaggle). After `make submit`, note "
              f"the public score next to this run (submission sha256 {digest}).")


if __name__ == "__main__":
    main()

"""LightGBM baseline for Kaggle Playground Series S6E10 (Predicting Airline Satisfaction).

Adapted from "Airline Satisfaction LightGBM Baseline AUC 0.9578" by Ali Hassan
https://www.kaggle.com/code/alikalwar/airline-satisfaction-lightgbm-baseline-auc-0-9578
Licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
Changes: Kaggle input paths replaced by a local data directory, wrapped in a function,
argparse CLI added, submission written next to the script's output directory.

Approach: object columns -> native categoricals, 3-fold stratified CV, averaged test
predictions. Metric: ROC AUC. Competition data is read from data/ and never committed.

Usage:  python training/baseline_lightgbm.py [--data-dir data] [--out-dir .]
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "satisfaction"
LGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.12,
    num_leaves=31,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    max_bin=127,
    verbose=-1,
    n_jobs=4,
)


def run(data_dir: Path, out_dir: Path, n_splits: int = 3, seed: int = 42) -> float:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    id_col = sample.columns[0]

    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, id_col])
    X_test = test.drop(columns=[id_col])

    # Encode categoricals on train+test together so both share the same category set.
    cats = list(X_test.select_dtypes(exclude="number").columns)  # object (pandas 2) or str (pandas 3)
    both = pd.concat([X, X_test], ignore_index=True)
    both[cats] = both[cats].astype("category")
    X = both.iloc[: len(train)].reset_index(drop=True)
    X_test = both.iloc[len(train):].reset_index(drop=True)

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / n_splits
        print(f"fold {fold} AUC {roc_auc_score(y[va_idx], oof[va_idx]):.5f}")

    cv_auc = roc_auc_score(y, oof)
    print(f"CV AUC {cv_auc:.5f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({id_col: test[id_col].values, sample.columns[1]: test_pred}).to_csv(
        out_dir / "submission.csv", index=False
    )
    return cv_auc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args()
    run(args.data_dir, args.out_dir)

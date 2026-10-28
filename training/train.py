"""Step 3a – Train the satisfaction model (PyTorch MLP, or a scikit-learn model for CT in step 8).

Reads the clean files from step 2 (data/processed/train.csv and validation.csv), trains on
train, measures on validation, and saves ONE file, artifacts/model.pt, holding the network
weights plus everything needed to preprocess new passengers (feature order, mean/std, metrics).
With params["model"] set to a SKLEARN_MODELS name the file is model.joblib instead (same contents).

    python -m training.train            # or: make train

Step 3b (MLflow) calls run_training() and logs / registers what it returns.
The API (step 4) loads the file with load_bundle() and calls predict_proba(), so training and
serving use the same preprocessing code.

Honest numbers: mean/std come from the training rows only, the number of epochs is fixed, and
the validation rows are never used to choose anything. The saved model is the exact model that
produced the reported metrics.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch import nn

from skysat import schema
from skysat.data import PROCESSED_DIR

OUT_DIR = Path("artifacts")
LOG_COLUMNS = ["flight_distance", "departure_delay_minutes", "arrival_delay_minutes"]  # long right tail
SCALED_COLUMNS = schema.NUMERIC_FEATURES + schema.RATING_FEATURES

DEFAULT_PARAMS = {
    "model": "mlp",            # or one of SKLEARN_MODELS (step 8: continuous training compares them)
    "epochs": 20,
    "batch_size": 512,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "hidden": [128, 64],
    "dropout": 0.1,
    "seed": 42,
    # Input choices suggested by `make graph` (see training/analysis/eda.py); the defaults keep the old input.
    "rating_one_hot": False,   # also give every 0-5 rating as separate yes/no columns (ratings act in steps, not a line)
    "drop_features": [],       # schema feature names to leave out, e.g. ["gender", "departure_delay_minutes"]
}
PREP_KEYS = ("rating_one_hot", "drop_features")  # the params that change to_matrix; saved with the model

# The other model types, trained on the same matrix as the MLP. The MLP-only params (epochs, lr, ...) are ignored.
SKLEARN_MODELS = {
    "logistic_regression": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    "random_forest": lambda seed: RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=-1,
                                                         random_state=seed),
    "gradient_boosting": lambda seed: HistGradientBoostingClassifier(random_state=seed),
    # Tuned by a 72-setting random search scored with 5-fold CV on train.csv only (validation.csv not used).
    "gradient_boosting_tuned": lambda seed: HistGradientBoostingClassifier(
        max_iter=1000, learning_rate=0.02, max_leaf_nodes=127, min_samples_leaf=20, l2_regularization=10.0,
        early_stopping=False, random_state=seed),
    "lightgbm": lambda seed: _lightgbm(seed),
    "xgboost": lambda seed: _xgboost(seed),
}


# LightGBM and XGBoost are imported only when one is trained, so code that never uses them does not need them.
# macOS: after PyTorch has run, multi-threaded LightGBM / XGBoost crash the process (two OpenMP runtimes, see
# training/__init__.py), so they train on one thread there. Linux (CI, GitHub CT) uses every core.
_BOOST_JOBS = 1 if sys.platform == "darwin" else -1
# Common library-default-style settings, fixed before training; no early stopping (that would tune on validation).
def _lightgbm(seed: int):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, n_jobs=_BOOST_JOBS, random_state=seed, verbose=-1)


def _xgboost(seed: int):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                         tree_method="hist", n_jobs=_BOOST_JOBS, random_state=seed)


def to_matrix(df: pd.DataFrame, rating_one_hot: bool = False, drop_features=()) -> np.ndarray:
    """Passenger rows (schema.FEATURES columns) -> float32 matrix: yes/no columns first, then scaled numbers.

    Deterministic: the one-hot columns come from the fixed category lists in schema.py, so a value
    we have never seen simply gives all zeros instead of changing the column layout.
    rating_one_hot adds 6 yes/no columns (0..5) per rating; drop_features removes columns by name.
    The scaled numbers are always the LAST columns (_standardize relies on that).
    """
    drop = set(drop_features)
    unknown = drop - set(schema.FEATURES)
    if unknown:
        raise ValueError(f"drop_features has unknown names: {sorted(unknown)}")

    one_hot = [
        (df[col].to_numpy()[:, None] == np.array(values)[None, :]).astype(np.float32)
        for col, values in schema.CATEGORICAL_VALUES.items() if col not in drop
    ]
    if rating_one_hot:
        levels = np.arange(schema.RATING_MIN, schema.RATING_MAX + 1)
        one_hot += [
            (df[col].to_numpy()[:, None] == levels[None, :]).astype(np.float32)
            for col in schema.RATING_FEATURES if col not in drop
        ]
    scaled = [c for c in SCALED_COLUMNS if c not in drop]
    numbers = df[scaled].astype(float)
    log_cols = [c for c in LOG_COLUMNS if c in scaled]
    numbers[log_cols] = np.log1p(numbers[log_cols].clip(lower=0))
    return np.hstack(one_hot + [numbers.to_numpy(dtype=np.float32)])


def _prep(params: dict) -> dict:
    """The to_matrix options stored in a params dict (older models have none -> old behaviour)."""
    return {k: params[k] for k in PREP_KEYS if k in params}


def n_default_inputs() -> int:
    return sum(len(v) for v in schema.CATEGORICAL_VALUES.values()) + len(SCALED_COLUMNS)


def make_model(n_in: int, hidden: list[int], dropout: float) -> nn.Module:
    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def pick_device() -> torch.device:
    """NVIDIA GPU, Apple GPU (Mac), else CPU (e.g. inside Docker)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _predict(model: nn.Module, X: torch.Tensor, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(X[i:i + batch_size])).squeeze(1) for i in range(0, len(X), batch_size)]
    return torch.cat(out).cpu().numpy()


def _standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = X.copy()
    n = len(mean)  # the scaled numbers are the last n columns
    X[:, -n:] = (X[:, -n:] - mean) / std
    return X


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Validation metrics at threshold 0.5, with the class balance and a majority-class baseline."""
    pred = (p >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "accuracy": float((pred == y).mean()),
        "positive_rate": float(y.mean()),
        "majority_baseline_accuracy": float(max(y.mean(), 1 - y.mean())),
    }


def run_training(train_df: pd.DataFrame, val_df: pd.DataFrame, params: dict | None = None,
                 out_dir: Path = OUT_DIR) -> tuple[str, dict]:
    """Train on train_df, measure on val_df, save out_dir/model.pt (or model.joblib). Returns (path, metrics)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if p["model"] != "mlp" and p["model"] not in SKLEARN_MODELS:
        raise ValueError(f"Unknown model {p['model']!r}; use 'mlp' or one of {sorted(SKLEARN_MODELS)}")
    torch.manual_seed(p["seed"])
    np.random.seed(p["seed"])
    device = pick_device()

    prep = _prep(p)
    X_train, X_val = to_matrix(train_df, **prep), to_matrix(val_df, **prep)
    y_train = train_df[schema.TARGET].to_numpy(dtype=np.float32)
    y_val = val_df[schema.TARGET].to_numpy(dtype=int)

    # Scaling statistics come from the training rows only.
    n_scaled = len([c for c in SCALED_COLUMNS if c not in p["drop_features"]])
    mean = X_train[:, -n_scaled:].mean(0)
    std = X_train[:, -n_scaled:].std(0) + 1e-6
    meta = {
        "features": schema.FEATURES, "mean": mean.tolist(), "std": std.tolist(), "params": p,
        "n_train_rows": len(train_df), "n_val_rows": len(val_df),
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    if p["model"] in SKLEARN_MODELS:
        print(f"model: {p['model']} | train rows: {len(X_train):,} | validation rows: {len(X_val):,} "
              f"| features: {X_train.shape[1]}")
        estimator = SKLEARN_MODELS[p["model"]](p["seed"]).fit(_standardize(X_train, mean, std), y_train)
        if "n_jobs" in estimator.get_params():
            estimator.set_params(n_jobs=1)  # serving: one CPU core per API copy, like torch.set_num_threads(1)
        meta["metrics"] = _metrics(y_val, estimator.predict_proba(_standardize(X_val, mean, std))[:, 1])
        path = out_dir / "model.joblib"
        joblib.dump({"model": estimator, "meta": meta}, path)
        return str(path), meta["metrics"]

    Xt = torch.from_numpy(_standardize(X_train, mean, std)).to(device)
    Xv = torch.from_numpy(_standardize(X_val, mean, std)).to(device)
    yt = torch.from_numpy(y_train).to(device)

    model = make_model(Xt.shape[1], p["hidden"], p["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    steps = (len(Xt) + p["batch_size"] - 1) // p["batch_size"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p["lr"], epochs=p["epochs"], steps_per_epoch=steps)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"device: {device} | train rows: {len(Xt):,} | validation rows: {len(Xv):,} | features: {Xt.shape[1]}")
    for epoch in range(p["epochs"]):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        total = 0.0
        for i in range(0, len(Xt), p["batch_size"]):
            idx = perm[i:i + p["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]).squeeze(1), yt[idx])
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        print(f"  epoch {epoch + 1}/{p['epochs']}  train loss {total / len(Xt):.4f}")

    metrics = _metrics(y_val, _predict(model, Xv))

    meta.update({"hidden": p["hidden"], "dropout": p["dropout"], "n_in": int(Xt.shape[1]), "metrics": metrics})
    path = out_dir / "model.pt"
    torch.save({"state_dict": model.cpu().state_dict(), "meta": meta}, path)
    return str(path), metrics


def model_file(folder) -> Path:
    """The model file inside a folder downloaded from MLflow: model.pt (MLP) or model.joblib (scikit-learn)."""
    for name in ("model.pt", "model.joblib"):
        if (Path(folder) / name).exists():
            return Path(folder) / name
    raise FileNotFoundError(f"No model.pt or model.joblib in {folder}")


def load_bundle(path) -> dict:
    """Load model.pt or model.joblib (model + preprocessing metadata) onto the CPU, ready for predict_proba."""
    if Path(path).suffix == ".joblib":
        return joblib.load(path)  # a pickle: only load files from our own MLflow registry
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    meta = bundle["meta"]
    n_in = meta.get("n_in", n_default_inputs())  # models saved before "n_in" existed used the default input
    model = make_model(n_in, meta["hidden"], meta["dropout"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return {"model": model, "meta": meta}


def predict_proba(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """P(satisfied) for passenger rows with the schema.FEATURES columns (extra columns are ignored)."""
    meta = bundle["meta"]
    X = _standardize(to_matrix(df[meta["features"]], **_prep(meta.get("params", {}))),
                     np.array(meta["mean"], dtype=np.float32), np.array(meta["std"], dtype=np.float32))
    if isinstance(bundle["model"], nn.Module):
        return _predict(bundle["model"], torch.from_numpy(X))
    return bundle["model"].predict_proba(X)[:, 1]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"ERROR: {path} not found. Run `make data` (python -m training.prepare_data) first.")
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser(description="Train the SkySat satisfaction model")
    ap.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--epochs", type=int, default=DEFAULT_PARAMS["epochs"])
    ap.add_argument("--lr", type=float, default=DEFAULT_PARAMS["lr"])
    ap.add_argument("--seed", type=int, default=DEFAULT_PARAMS["seed"])
    ap.add_argument("--rating-one-hot", action="store_true", help="add one-hot columns for every 0-5 rating")
    ap.add_argument("--drop-features", nargs="*", default=[], metavar="NAME",
                    help="schema feature names to leave out, e.g. gender departure_delay_minutes")
    args = ap.parse_args()

    train_df, val_df = _read(args.data_dir / "train.csv"), _read(args.data_dir / "validation.csv")
    path, metrics = run_training(train_df, val_df, {"epochs": args.epochs, "lr": args.lr, "seed": args.seed,
                                  "rating_one_hot": args.rating_one_hot, "drop_features": args.drop_features},
                                 args.out_dir)
    print("validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("saved", path)


if __name__ == "__main__":
    main()

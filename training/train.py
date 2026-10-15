"""Step 3a – Train the satisfaction model (PyTorch MLP).

Reads the clean files from step 2 (data/processed/train.csv and validation.csv), trains on
train, measures on validation, and saves ONE file, artifacts/model.pt, holding the network
weights plus everything needed to preprocess new passengers (feature order, mean/std, metrics).

    python -m training.train            # or: make train

Step 3b (MLflow) calls run_training() and logs / registers what it returns.
The API (step 4) loads the file with load_bundle() and calls predict_proba(), so training and
serving use the same preprocessing code.

Honest numbers: mean/std come from the training rows only, the number of epochs is fixed, and
the validation rows are never used to choose anything. The saved model is the exact model that
produced the reported metrics.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch import nn

from skysat import schema
from skysat.data import PROCESSED_DIR

OUT_DIR = Path("artifacts")
LOG_COLUMNS = ["flight_distance", "departure_delay_minutes", "arrival_delay_minutes"]  # long right tail
SCALED_COLUMNS = schema.NUMERIC_FEATURES + schema.RATING_FEATURES

DEFAULT_PARAMS = {
    "epochs": 20,
    "batch_size": 512,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "hidden": [128, 64],
    "dropout": 0.1,
    "seed": 42,
}


def to_matrix(df: pd.DataFrame) -> np.ndarray:
    """Passenger rows (schema.FEATURES columns) -> float32 matrix: one-hot text columns, then numbers.

    Deterministic: the one-hot columns come from the fixed category lists in schema.py, so a value
    we have never seen simply gives all zeros instead of changing the column layout.
    """
    one_hot = [
        (df[col].to_numpy()[:, None] == np.array(values)[None, :]).astype(np.float32)
        for col, values in schema.CATEGORICAL_VALUES.items()
    ]
    numbers = df[SCALED_COLUMNS].astype(float)
    numbers[LOG_COLUMNS] = np.log1p(numbers[LOG_COLUMNS].clip(lower=0))
    return np.hstack(one_hot + [numbers.to_numpy(dtype=np.float32)])


def n_one_hot() -> int:
    return sum(len(v) for v in schema.CATEGORICAL_VALUES.values())


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
    X[:, n_one_hot():] = (X[:, n_one_hot():] - mean) / std
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
    """Train on train_df, measure on val_df, save out_dir/model.pt. Returns (model path, metrics)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    torch.manual_seed(p["seed"])
    np.random.seed(p["seed"])
    device = pick_device()

    X_train, X_val = to_matrix(train_df), to_matrix(val_df)
    y_train = train_df[schema.TARGET].to_numpy(dtype=np.float32)
    y_val = val_df[schema.TARGET].to_numpy(dtype=int)

    # Scaling statistics come from the training rows only.
    mean = X_train[:, n_one_hot():].mean(0)
    std = X_train[:, n_one_hot():].std(0) + 1e-6
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

    meta = {
        "features": schema.FEATURES, "mean": mean.tolist(), "std": std.tolist(),
        "hidden": p["hidden"], "dropout": p["dropout"], "params": p,
        "n_train_rows": len(train_df), "n_val_rows": len(val_df), "metrics": metrics,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model.pt"
    torch.save({"state_dict": model.cpu().state_dict(), "meta": meta}, path)
    return str(path), metrics


def load_bundle(path) -> dict:
    """Load model.pt (weights + preprocessing metadata) onto the CPU, ready for predict_proba."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    meta = bundle["meta"]
    model = make_model(n_one_hot() + len(SCALED_COLUMNS), meta["hidden"], meta["dropout"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return {"model": model, "meta": meta}


def predict_proba(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """P(satisfied) for passenger rows with the schema.FEATURES columns (extra columns are ignored)."""
    meta = bundle["meta"]
    X = _standardize(to_matrix(df[meta["features"]]),
                     np.array(meta["mean"], dtype=np.float32), np.array(meta["std"], dtype=np.float32))
    return _predict(bundle["model"], torch.from_numpy(X))


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
    args = ap.parse_args()

    train_df, val_df = _read(args.data_dir / "train.csv"), _read(args.data_dir / "validation.csv")
    path, metrics = run_training(train_df, val_df, {"epochs": args.epochs, "lr": args.lr, "seed": args.seed},
                                 args.out_dir)
    print("validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("saved", path)


if __name__ == "__main__":
    main()

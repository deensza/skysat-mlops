"""PyTorch MLP for Kaggle Playground Series S6E10 (Predicting Airline Satisfaction).

Same data handling and CV protocol as baseline_lightgbm.py (itself adapted from Ali Hassan's
Apache 2.0 notebook, https://www.kaggle.com/code/alikalwar/airline-satisfaction-lightgbm-baseline-auc-0-9578):
3-fold stratified CV (seed 42), ROC AUC, averaged test predictions -> submission.csv.
This is a plain, untuned MLP; hyperparameters are exposed on the CLI.

Preprocessing statistics (mean/std) are fitted on each training fold only. The number of
epochs is fixed, so the validation fold is never used to pick a checkpoint and the OOF
metrics are an honest estimate.

Output: submission.csv and model.pt. model.pt is ONE file holding the network weights plus
everything needed to preprocess new rows (column list, mean/std, metrics). Training and
serving use the same `preprocess()` / `predict_proba()` functions below, so they cannot drift.

Runs anywhere: picks cuda / mps (Mac) / cpu automatically, and finds the data in --data-dir
(local, Docker volume, mounted Drive) or in /kaggle/input when run as a Kaggle notebook.
Self-contained on purpose (a Kaggle script kernel is a single file).

Usage:  python training/train_torch.py [--data-dir data] [--out-dir .] [--epochs 15]
Serve:  bundle = load_bundle("model.pt"); predict_proba(bundle, pandas_dataframe_of_raw_rows)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn

TARGET = "satisfaction"
DELAY_COLS = ["Departure Delay in Minutes", "Arrival Delay in Minutes"]


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw feature rows (no id / target) -> all-numeric float32 frame. Used by training AND serving."""
    df = raw.copy()
    # Arrival delay is the only column with missing values; keep a flag and fill from departure delay.
    df["arrival_delay_missing"] = df[DELAY_COLS[1]].isna().astype(np.float32)
    df[DELAY_COLS[1]] = df[DELAY_COLS[1]].fillna(df[DELAY_COLS[0]])
    for c in DELAY_COLS + ["Flight Distance"]:
        df[c] = np.log1p(df[c].clip(lower=0))
    cats = list(df.select_dtypes(exclude="number").columns)  # object (pandas 2) or str (pandas 3)
    return pd.get_dummies(df, columns=cats, dtype=np.float32).astype(np.float32)


def to_matrix(raw: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Preprocess and align to the training column list (unseen categories -> 0, missing -> 0)."""
    return preprocess(raw).reindex(columns=columns, fill_value=0.0).to_numpy(dtype=np.float32)


def make_model(n_in: int, hidden: list[int], dropout: float) -> nn.Module:
    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def pick_device() -> torch.device:
    """NVIDIA GPU (Linux/Colab/Kaggle), Apple GPU (Mac), else CPU (e.g. inside Docker)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_data_dir(data_dir: Path) -> Path:
    """Use --data-dir if it has train.csv; otherwise look in Kaggle's read-only input folder."""
    if (data_dir / "train.csv").exists():
        return data_dir
    hits = sorted(Path("/kaggle/input").glob("**/train.csv"))
    if hits:
        return hits[0].parent
    raise FileNotFoundError(f"train.csv not found in {data_dir} or /kaggle/input")


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(X[i:i + batch_size])).squeeze(1) for i in range(0, len(X), batch_size)]
    return torch.cat(out).cpu().numpy()


def fit_fold(X_tr, y_tr, args, device) -> nn.Module:
    model = make_model(X_tr.shape[1], args.hidden, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = (len(X_tr) + args.batch_size - 1) // args.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
        total = 0.0
        for i in range(0, len(X_tr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X_tr[idx]).squeeze(1), y_tr[idx])
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        print(f"  epoch {epoch + 1}/{args.epochs} train loss {total / len(X_tr):.4f}")
    return model


def load_bundle(path) -> dict:
    """Load model.pt (weights + preprocessing metadata) onto the CPU, ready for predict_proba."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    meta = bundle["meta"]
    model = make_model(len(meta["columns"]), meta["hidden"], meta["dropout"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return {"model": model, "meta": meta}


def predict_proba(bundle: dict, raw: pd.DataFrame) -> np.ndarray:
    """P(satisfied) for raw feature rows (the competition columns minus id / satisfaction)."""
    meta = bundle["meta"]
    X = to_matrix(raw, meta["columns"])
    sc = meta["scale_cols"]
    X[:, sc] = (X[:, sc] - np.array(meta["mean"], dtype=np.float32)) / np.array(meta["std"], dtype=np.float32)
    return predict(bundle["model"], torch.from_numpy(X), 4096)


def report_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Out-of-fold metrics at threshold 0.5, with the class balance and a majority-class baseline."""
    pred = (p >= 0.5).astype(int)
    m = {
        "roc_auc": float(roc_auc_score(y, p)),
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "accuracy": float((pred == y).mean()),
        "positive_rate": float(y.mean()),
        "majority_baseline_accuracy": float(max(y.mean(), 1 - y.mean())),
    }
    for k, v in m.items():
        print(f"  {k}: {v:.5f}")
    return m


def run(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    data_dir = find_data_dir(args.data_dir)
    print("device:", device, "| data:", data_dir)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    id_col = sample.columns[0]
    print("rows: train", len(train), "test", len(test), "| missing values:", int(train.isna().sum().sum()))

    y = train[TARGET].astype(int).values
    raw_train = train.drop(columns=[TARGET, id_col])
    raw_test = test.drop(columns=[id_col])
    # One-hot on train+test together so both share the same column set (uses no target information).
    both = preprocess(pd.concat([raw_train, raw_test], ignore_index=True))
    columns = list(both.columns)
    scale_cols = [i for i, c in enumerate(columns) if both[c].nunique() > 2]  # skip one-hot / flag columns
    X, X_test = both.to_numpy()[: len(train)], both.to_numpy()[len(train):]

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(X_test))
    skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        mean = X[tr_idx][:, scale_cols].mean(0)
        std = X[tr_idx][:, scale_cols].std(0) + 1e-6

        def prep(a):
            a = a.copy()
            a[:, scale_cols] = (a[:, scale_cols] - mean) / std
            return torch.from_numpy(a).to(device)

        X_tr, X_va, X_te = prep(X[tr_idx]), prep(X[va_idx]), prep(X_test)
        y_tr = torch.from_numpy(y[tr_idx].astype(np.float32)).to(device)

        print(f"fold {fold}")
        model = fit_fold(X_tr, y_tr, args, device)
        oof[va_idx] = predict(model, X_va, args.batch_size * 4)
        test_pred += predict(model, X_te, args.batch_size * 4) / args.folds
        print(f"fold {fold} AUC {roc_auc_score(y[va_idx], oof[va_idx]):.5f}")

    print("out-of-fold metrics:")
    metrics = report_metrics(y, oof)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({id_col: test[id_col].values, sample.columns[1]: test_pred}).to_csv(
        args.out_dir / "submission.csv", index=False)

    # The CV models above only measured the score. Fit one final model on all the data and
    # save it as a single bundle with everything needed to preprocess new rows.
    print("final fit on all training rows")
    mean = X[:, scale_cols].mean(0)
    std = X[:, scale_cols].std(0) + 1e-6
    X_all = X.copy()
    X_all[:, scale_cols] = (X_all[:, scale_cols] - mean) / std
    final = fit_fold(torch.from_numpy(X_all).to(device),
                     torch.from_numpy(y.astype(np.float32)).to(device), args, device)
    meta = {"columns": columns, "scale_cols": scale_cols, "mean": mean.tolist(), "std": std.tolist(),
            "hidden": args.hidden, "dropout": args.dropout, "seed": args.seed,
            "n_train_rows": len(train), "metrics": metrics}
    torch.save({"state_dict": final.cpu().state_dict(), "meta": meta}, args.out_dir / "model.pt")
    print("saved", args.out_dir / "model.pt")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    run(ap.parse_args())

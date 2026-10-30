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

2026-10-30, same folds: LightGBM (LGB_PARAMS, early stopping on the last 10% of each training fold) beat the HGB
above in all 5 folds, and the average of both models' logits beat both:
    hgb 0.95852 | lightgbm 0.95899 | average of logits 0.95904
Then inputs from the original dataset (external, rule 6.a): for every column, the share of satisfied passengers
with the same value there, overall and within the same type of travel and class (original_rates):
    hgb 0.95966 | lightgbm 0.96012 | average of logits 0.96013 ± 0.00053  <- default; better in all 5 folds
Screened on fold 1 and not kept: every column as a category 0.95957, value counts 0.95971, rates + categories
0.95974 (vs. rates alone 0.96021). With the rates, averaging adds almost nothing over LightGBM alone.

Each run logs to MLflow experiment `skysat-kaggle`: CV per model and fold, out-of-fold plots (ROC, calibration,
threshold, confusion, folds, input importance), tables and report.md. Aggregates only, never competition rows.
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
LGB_PARAMS = {"n_estimators": 3000, "learning_rate": 0.02, "num_leaves": 63, "min_child_samples": 40, "subsample": 0.8,
              "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 5.0}
MODELS = ["hgb", "lightgbm"]


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


RATE_KEY = ["Type of Travel", "Class"]


def original_rates(df: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    """For every column: the share of satisfied passengers with the same value in the original dataset, overall and
    within the same type of travel and class. Uses only the original (external) labels, never the competition's."""
    out = {}
    for col in [c for c in original.columns if c not in ("id", TARGET)]:
        out[f"{col} rate"] = df[col].map(original.groupby(col)[TARGET].mean()).astype(float).to_numpy()
        if col not in RATE_KEY:
            rates = original.groupby(RATE_KEY + [col])[TARGET].mean().rename("r").reset_index()
            out[f"{col} rate by travel and class"] = df[RATE_KEY + [col]].merge(rates, how="left")["r"].to_numpy()
    return pd.DataFrame(out, index=df.index)


def features(df: pd.DataFrame, categories: dict, original: pd.DataFrame | None = None) -> pd.DataFrame:
    X = df.drop(columns=["id", TARGET], errors="ignore")
    if original is not None:
        X = pd.concat([X, original_rates(df, original)], axis=1)
    for col in CATEGORICAL:
        X[col] = X[col].astype(categories[col])  # same category codes in every frame
    return X


def model(params: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=SEED, **params)


def fit_predict(kind: str, params: dict, X_fit, y_fit, frames: list) -> tuple[list[np.ndarray], int, pd.Series | None]:
    """Train one model on the fit rows: P(satisfied) for each frame in frames, its number of rounds, and (LightGBM)
    the share of the total gain from splits on each input."""
    if kind == "hgb":
        fitted = model(params).fit(X_fit, y_fit)
        return [fitted.predict_proba(f)[:, 1] for f in frames], fitted.n_iter_, None
    import lightgbm as lgb
    cut = int(len(X_fit) * 0.9)  # early stopping on the last 10% of the training fold, like HGB's
    fitted = lgb.LGBMClassifier(**LGB_PARAMS, random_state=SEED, verbose=-1).fit(
        X_fit.iloc[:cut], y_fit.iloc[:cut], eval_X=X_fit.iloc[cut:], eval_y=y_fit.iloc[cut:], eval_metric="auc",
        callbacks=[lgb.early_stopping(200, verbose=False)])
    gain = pd.Series(fitted.booster_.feature_importance("gain"), index=X_fit.columns)
    return [fitted.predict_proba(f)[:, 1] for f in frames], fitted.best_iteration_, gain / gain.sum()


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cross_validate(train: pd.DataFrame, test: pd.DataFrame, extra: pd.DataFrame | None, params: dict,
                   folds: int = 5, models: list[str] = ("hgb",),
                   rates_from: pd.DataFrame | None = None) -> tuple[list[float], np.ndarray, dict]:
    """Per-fold ROC AUC on held-out competition rows, the test predictions averaged over the folds, and details for
    MLflow: out-of-fold predictions, every model's score per fold, LightGBM's input importance.
    With several models, each fold's prediction is the average of the models' logits."""
    frames = [train, test] + ([extra] if extra is not None else [])
    categories = {c: CategoricalDtype(sorted(pd.concat([f[c] for f in frames]).dropna().unique())) for c in CATEGORICAL}
    X, y, X_test = features(train, categories, rates_from), train[TARGET], features(test, categories, rates_from)
    X_extra = features(extra, categories, rates_from) if extra is not None else None

    scores, test_pred, oof, rows, gains = [], np.zeros(len(test)), np.zeros(len(train)), [], []
    for k, (fit_idx, val_idx) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, y)):
        X_fit, y_fit = X.iloc[fit_idx], y.iloc[fit_idx]
        if X_extra is not None:  # external rows only ever train; they are never scored
            X_fit, y_fit = pd.concat([X_fit, X_extra]), pd.concat([y_fit, extra[TARGET]])
        val_logits, test_logits, parts = [], [], []
        for kind in models:
            (p_val, p_test), rounds, gain = fit_predict(kind, params, X_fit, y_fit, [X.iloc[val_idx], X_test])
            val_logits.append(logit(p_val))
            test_logits.append(logit(p_test))
            rows.append({"fold": k + 1, "model": kind, "roc_auc": roc_auc_score(y.iloc[val_idx], p_val), "rounds": rounds})
            parts.append(f"{kind} {rows[-1]['roc_auc']:.5f} ({rounds} rounds)")
            if gain is not None:
                gains.append(gain)
        p_val, p_test = (1 / (1 + np.exp(-np.mean(lg, axis=0))) for lg in (val_logits, test_logits))
        scores.append(roc_auc_score(y.iloc[val_idx], p_val))
        rows.append({"fold": k + 1, "model": "+".join(models), "roc_auc": scores[-1], "rounds": None})
        oof[val_idx] = p_val
        test_pred += p_test / folds
        print(f"  fold {k + 1}/{folds}: roc_auc {scores[-1]:.5f} | " + ", ".join(parts), flush=True)
    importance = pd.concat(gains, axis=1).mean(axis=1).sort_values(ascending=False) if gains else None
    return scores, test_pred, {"oof": oof, "folds": pd.DataFrame(rows), "importance": importance}


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
    ap.add_argument("--no-original-rates", dest="original_rates", action="store_false",
                    help="leave out the original dataset's satisfied share per value (CV 0.95905 instead of 0.96013)")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=MODELS, help="several = average of their logits")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-mlflow", action="store_true", help="do not log the run (quick experiments)")
    args = ap.parse_args()

    train, test = load_competition()
    extra = load_original(list(train.columns)) if args.original else None
    rates_from = load_original(list(train.columns)) if args.original_rates else None
    params = {"learning_rate": args.learning_rate, "max_iter": args.max_iter, "max_leaf_nodes": args.max_leaf_nodes}
    print(f"competition train {len(train):,} rows ({train[TARGET].mean():.1%} satisfied), test {len(test):,} rows, "
          f"original added: {0 if extra is None else len(extra):,} rows | {args.models} | hgb {params}")

    scores, pred, details = cross_validate(train, test, extra, params, args.folds, args.models, rates_from)
    path = write_submission(test, pred)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    print(f"CV roc_auc {np.mean(scores):.5f} ± {np.std(scores):.5f} | wrote {path.relative_to(PROJECT_ROOT)} "
          f"(sha256 {digest})")

    if not args.no_mlflow:
        settings = {**params, "model": "+".join(args.models), "folds": args.folds, "train_rows": len(train),
                    "original_rows": 0 if extra is None else len(extra), "original_rates": args.original_rates,
                    **({f"lgb_{k}": v for k, v in LGB_PARAMS.items()} if "lightgbm" in args.models else {})}
        log_run(settings, scores, details, train[TARGET].to_numpy(), digest)


def _report(settings: dict, scores: list[float], details: dict, digest: str) -> str:
    folds = details["folds"].pivot(index="fold", columns="model", values="roc_auc").round(5)
    lines = [f"# Kaggle run: {settings['model']}", "",
             f"**5-fold CV ROC AUC {np.mean(scores):.5f} ± {np.std(scores):.5f}** (out-of-fold, competition train rows)",
             f"· submission sha256 `{digest}` · after `make submit`, the public score is in `make kaggle-scores`", "",
             "## Every model, every fold", "", "| fold | " + " | ".join(folds.columns) + " |",
             "|---|" + "---|" * len(folds.columns)]
    lines += [f"| {f} | " + " | ".join(f"{v:.5f}" for v in row) + " |" for f, row in folds.iterrows()]
    lines += [f"| mean | " + " | ".join(f"{v:.5f}" for v in folds.mean()) + " |", ""]
    if details["importance"] is not None:
        lines += ["## Strongest inputs (LightGBM, share of gain)", ""]
        lines += [f"- {name}: {share:.1%}" for name, share in details["importance"].head(8).items()]
        lines.append("")
    lines += ["## Settings", ""] + [f"- {k}: {v}" for k, v in settings.items()]
    return "\n".join(lines) + "\n"


def log_run(settings: dict, scores: list[float], details: dict, y: np.ndarray, digest: str) -> None:
    """One run in experiment `skysat-kaggle`: settings, CV scores per model and fold, plots and a report.
    Aggregates only (scores, curves, counts): no competition rows and no submission file leave the PC."""
    import mlflow

    from training.track import git_commit
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or "http://localhost:5050")
    mlflow.set_experiment("skysat-kaggle")
    folds, oof = details["folds"], details["oof"]
    with mlflow.start_run(tags={"purpose": "kaggle", "git_commit": git_commit(), "submission_sha256": digest}):
        mlflow.log_params(settings)
        metrics = {"cv_roc_auc_mean": float(np.mean(scores)), "cv_roc_auc_std": float(np.std(scores)),
                   "oof_roc_auc": roc_auc_score(y, oof), "oof_accuracy": float(((oof >= 0.5) == y).mean()),
                   **{f"cv_roc_auc_fold{k + 1}": s for k, s in enumerate(scores)}}
        for m, g in folds.groupby("model"):
            if m != settings["model"]:
                metrics[f"{m}.cv_roc_auc_mean"] = float(g["roc_auc"].mean())
        mlflow.log_metrics(metrics)
        mlflow.log_text(folds.round(5).to_csv(index=False), "tables/folds.csv")
        if details["importance"] is not None:
            mlflow.log_text(details["importance"].round(5).rename("gain_share").to_csv(), "tables/feature_importance.csv")
        mlflow.log_text(_report(settings, scores, details, digest), "report.md")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed: plots skipped")
        else:
            _plots(plt, y, oof, folds, details["importance"], settings["model"])
    print(f"Logged to MLflow ({mlflow.get_tracking_uri()}, experiment skysat-kaggle): report.md, plots/, tables/. "
          f"After `make submit`, note the public score next to this run (submission sha256 {digest}).")


def draw_folds(ax, folds: pd.DataFrame) -> None:
    from training.insight import BLUE, GREEN, GREY, ORANGE, _caption
    table = folds.pivot(index="fold", columns="model", values="roc_auc")
    width = 0.8 / len(table.columns)
    for i, (m, color) in enumerate(zip(table.columns, [BLUE, ORANGE, GREEN, GREY])):
        ax.bar(table.index + (i - (len(table.columns) - 1) / 2) * width, table[m], width, color=color,
               label=f"{m} (mean {table[m].mean():.5f})")
    ax.set(xlabel="fold", ylabel="ROC AUC on the fold's held-out rows", title="Every model on every fold",
           ylim=(table.min().min() - 0.0015, table.max().max() + 0.001), xticks=table.index)
    ax.legend(fontsize=7, loc="lower right")
    _caption(ax, "The combination should beat each model in every fold, not only on average: then the gain is real.")


def draw_gain(ax, importance: pd.Series, top: int = 15) -> None:
    from training.insight import BLUE, ORANGE, _caption
    d = importance.head(top)[::-1]
    ax.barh(d.index, d.values, color=[ORANGE if " rate" in n else BLUE for n in d.index])
    ax.set(xlabel="share of LightGBM's total gain", title=f"Strongest inputs (top {top})")
    ax.tick_params(axis="y", labelsize=6.5)
    _caption(ax, "Orange: rates taken from the original dataset. Blue: the competition's own columns.")


def _plots(plt, y, oof, folds, importance, model_name) -> None:
    import mlflow

    from training import insight
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                         "axes.titlesize": 10, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    views = [("folds", lambda ax: draw_folds(ax, folds)), ("roc_curve", lambda ax: insight.draw_roc(ax, y, oof)),
             ("calibration", lambda ax: insight.draw_calibration(ax, y, oof)),
             ("threshold", lambda ax: insight.draw_threshold(ax, y, oof)),
             ("confusion_matrix", lambda ax: insight.draw_confusion(ax, y, oof))]
    if importance is not None:
        views.append(("feature_importance", lambda ax: draw_gain(ax, importance)))
    for name, draw in views:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        draw(ax)
        fig.tight_layout()
        mlflow.log_figure(fig, f"plots/{name}.png")
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, (_, draw) in zip(axes.flat, views):
        draw(ax)
    for ax in axes.flat[len(views):]:
        ax.axis("off")
    fig.suptitle(f"Kaggle {model_name}: out-of-fold predictions on all {len(y):,} competition train rows",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/overview.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Extra detail for an MLflow run, at no extra runs: the model's real settings, training curves, plots and tables.

    settings = model_settings(bundle)        # {"model.n_estimators": 500, ...}; ensembles: per member
    log_insights(bundle, val_df)             # inside an active run: everything below, on the validation rows

What a run gets (artifacts; aggregates only, never passenger rows: the competition data must not be shared):

    report.md                        one-page summary: scores, data, settings, the strongest features
    plots/roc_curve.png              how well it ranks satisfied above not satisfied
    plots/precision_recall.png       the trade-off when choosing a threshold
    plots/calibration.png            are the probabilities honest? (predicted vs. real share, and their spread)
    plots/confusion_matrix.png       right and wrong answers at threshold 0.5
    plots/feature_importance.png     how much ROC AUC drops when one input is shuffled
    plots/train_loss.png             the MLP's loss per epoch (MLP and ensembles)
    tables/feature_importance.csv    the numbers behind the plot
    tables/metrics_by_group.csv      the scores per passenger group (class, travel type, ...): where is it weak?

and metrics: `train_loss` per epoch (MLP), `<member>.train_loss` (ensembles). Called by training/ct.py.
Plots need matplotlib (in requirements.txt); without it only the metrics and tables are logged.
"""

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score,
                             roc_curve)

from skysat import schema
from training.train import predict_proba

MLP_ONLY = ("epochs", "batch_size", "lr", "weight_decay", "hidden", "dropout")  # ignored by the other models
IMPORTANCE_ROWS = 5000  # feature importance on a fixed sample of the validation rows: enough, and fast
GROUPS = list(schema.CATEGORICAL_VALUES)  # gender, customer_type, type_of_travel, travel_class


def run_params(params: dict) -> dict:
    """The training params that apply to this model type (the MLP-only ones are left out for the others)."""
    if params["model"] == "mlp":
        return params
    return {k: v for k, v in params.items() if k not in MLP_ONLY}


def model_settings(bundle: dict) -> dict:
    """The fitted estimator's own settings (trees, depth, ...), prefixed `model.`; ensembles: per member."""
    if "members" in bundle:
        return {f"{name}.{k}": v for name, member in bundle["members"].items()
                for k, v in (model_settings(member) or run_params(member["meta"]["params"])).items()}
    model = bundle["model"]
    if not hasattr(model, "get_params"):
        return {}
    simple = (str, int, float, bool, type(None))
    return {f"model.{k}": v if isinstance(v, simple) else str(v) for k, v in sorted(model.get_params().items())}


def loss_curves(bundle: dict) -> dict:
    """{metric name: loss per epoch}: `train_loss` for an MLP, `<member>.train_loss` inside an ensemble."""
    if "members" in bundle:
        return {f"{name}.train_loss": m["meta"]["train_loss"] for name, m in bundle["members"].items()
                if m["meta"].get("train_loss")}
    loss = bundle["meta"].get("train_loss")
    return {"train_loss": loss} if loss else {}


def feature_importance(bundle: dict, val_df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Permutation importance: ROC AUC drop when one input column is shuffled (works for every model type)."""
    rows = val_df.sample(min(IMPORTANCE_ROWS, len(val_df)), random_state=seed)
    y = rows[schema.TARGET].to_numpy(dtype=int)
    base = roc_auc_score(y, predict_proba(bundle, rows))
    rng = np.random.default_rng(seed)
    out = []
    for col in schema.FEATURES:
        shuffled = rows.copy()
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
        out.append({"feature": col, "roc_auc_drop": base - roc_auc_score(y, predict_proba(bundle, shuffled))})
    return pd.DataFrame(out).sort_values("roc_auc_drop", ascending=False, ignore_index=True)


def metrics_by_group(val_df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Scores per passenger group: a model can be great on average and weak for one group."""
    y, pred = val_df[schema.TARGET].to_numpy(dtype=int), (p >= 0.5).astype(int)
    out = []
    for col in GROUPS:
        for value in schema.CATEGORICAL_VALUES[col]:
            m = (val_df[col] == value).to_numpy()
            if m.sum() == 0:
                continue
            auc = roc_auc_score(y[m], p[m]) if len(set(y[m])) == 2 else float("nan")
            out.append({"group": col, "value": value, "passengers": int(m.sum()),
                        "satisfied_share": round(float(y[m].mean()), 4),
                        "accuracy": round(float((pred[m] == y[m]).mean()), 4), "roc_auc": round(float(auc), 4)})
    return pd.DataFrame(out)


def _report(bundle: dict, settings: dict, importance: pd.DataFrame, groups: pd.DataFrame) -> str:
    meta, metrics = bundle["meta"], bundle["meta"]["metrics"]
    weakest = groups.sort_values("accuracy").iloc[0]
    lines = [
        f"# Model report: {meta['params']['model']}", "",
        f"Trained on {meta['n_train_rows']:,} rows, scored on {meta['n_val_rows']:,} validation rows.", "",
        "| Metric | Value |", "|---|---|",
        *[f"| {k} | {v:.4f} |" for k, v in metrics.items()], "",
        "## Strongest inputs (ROC AUC drop when shuffled)", "",
        *[f"{i + 1}. {r.feature}: {r.roc_auc_drop:.4f}" for i, r in importance.head(5).iterrows()], "",
        f"Weakest passenger group: **{weakest['group']} = {weakest['value']}** "
        f"(accuracy {weakest['accuracy']:.4f} on {weakest['passengers']:,} passengers; overall "
        f"{metrics['accuracy']:.4f}). All groups: tables/metrics_by_group.csv", "",
        f"## Settings ({len(settings)})", "", *[f"- `{k}`: {v}" for k, v in settings.items()],
    ]
    return "\n".join(lines) + "\n"


def log_insights(bundle: dict, val_df: pd.DataFrame) -> None:
    """Inside an active MLflow run: loss curves, plots, tables and report.md (see the module docstring)."""
    run_id = mlflow.active_run().info.run_id
    curves = loss_curves(bundle)
    if curves:  # one request for all epochs (DagsHub is slow per request)
        from mlflow.entities import Metric
        now = int(pd.Timestamp.now().timestamp() * 1000)
        mlflow.MlflowClient().log_batch(run_id, metrics=[
            Metric(name, float(v), now, step) for name, loss in curves.items() for step, v in enumerate(loss, 1)])

    y = val_df[schema.TARGET].to_numpy(dtype=int)
    p = predict_proba(bundle, val_df)
    importance = feature_importance(bundle, val_df)
    groups = metrics_by_group(val_df, p)
    mlflow.log_text(importance.round(5).to_csv(index=False), "tables/feature_importance.csv")
    mlflow.log_text(groups.to_csv(index=False), "tables/metrics_by_group.csv")
    mlflow.log_text(_report(bundle, model_settings(bundle) or run_params(bundle["meta"]["params"]), importance,
                            groups), "report.md")

    try:
        import matplotlib
        matplotlib.use("Agg")  # no screen needed (CT runs on a server)
        import matplotlib.pyplot as plt
        import matplotlib.ticker
    except ImportError:
        print("matplotlib not installed: plots skipped")
        return

    def save(fig, name):
        fig.tight_layout()
        mlflow.log_figure(fig, f"plots/{name}.png")
        plt.close(fig)

    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC {roc_auc_score(y, p):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="random guess")
    ax.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curve (validation)")
    ax.legend(loc="lower right")
    save(fig, "roc_curve")

    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, label=f"average precision {average_precision_score(y, p):.4f}")
    ax.axhline(y.mean(), ls="--", color="grey", label=f"always 'satisfied' ({y.mean():.3f})")
    ax.set(xlabel="recall", ylabel="precision", title="Precision vs. recall (validation)", ylim=(0, 1.02))
    ax.legend(loc="lower left")
    save(fig, "precision_recall")

    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    used = [b for b in range(10) if (idx == b).any()]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5, 6), height_ratios=[3, 1.4], sharex=True)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfectly honest")
    ax.plot([p[idx == b].mean() for b in used], [y[idx == b].mean() for b in used], "o-", label="model")
    ax.set(ylabel="real share satisfied", title="Calibration (validation)")
    ax.legend(loc="upper left")
    ax2.hist([p[y == 0], p[y == 1]], bins=20, label=["not satisfied", "satisfied"], stacked=True)
    ax2.set(xlabel="predicted probability", ylabel="passengers")
    ax2.legend(fontsize=7)
    save(fig, "calibration")

    cm = confusion_matrix(y, (p >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    labels = ["not satisfied", "satisfied"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
           xlabel="predicted", ylabel="real answer", title="Confusion matrix (validation, threshold 0.5)")
    for (i, j), n in np.ndenumerate(cm):
        ax.text(j, i, f"{n:,}", ha="center", va="center", color="white" if n > cm.max() / 2 else "black")
    save(fig, "confusion_matrix")

    top = importance.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top["feature"], top["roc_auc_drop"])
    ax.set(xlabel="ROC AUC drop when shuffled", title=f"Feature importance ({IMPORTANCE_ROWS:,} validation rows)")
    save(fig, "feature_importance")

    if curves:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for name, loss in curves.items():
            ax.plot(range(1, len(loss) + 1), loss, "o-", label=name)
        ax.set(xlabel="epoch", ylabel="training loss", title="MLP training curve")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend()
        save(fig, "train_loss")

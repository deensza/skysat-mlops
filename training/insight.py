"""Extra detail for an MLflow run, at no extra runs: the model's real settings, training curves, plots and tables.

    settings = model_settings(bundle)        # {"model.n_estimators": 500, ...}; ensembles: per member
    log_insights(bundle, val_df)             # inside an active run: everything below, on the validation rows

What a run gets (aggregates only, never passenger rows: the competition data must not be shared):

    plots/overview.png               START HERE: the six views below on one page
    plots/roc_curve.png              how well it ranks satisfied above not satisfied
    plots/precision_recall.png       precision vs. recall
    plots/threshold.png              precision, recall, F1, accuracy for every decision threshold
    plots/calibration.png            are the probabilities honest? (predicted vs. real share, and their spread)
    plots/confusion_matrix.png       right and wrong answers at threshold 0.5, with percentages
    plots/feature_importance.png     how much ROC AUC drops when one input is shuffled
    plots/groups.png                 accuracy per passenger group (class, travel type, delay, age, ...)
    plots/train_loss.png             the MLP's loss per epoch (MLP and ensembles)
    tables/feature_importance.csv    the numbers behind the importance plot
    tables/metrics_by_group.csv      the numbers behind the group plot
    report.md                        one-page summary: scores, strongest inputs, weakest group, settings

and metrics: `train_loss` per epoch (MLP), `<member>.train_loss` (ensembles), `group.<column>.<value>.accuracy`
and `.roc_auc` (compare groups across runs in MLflow's compare view). Called by training/ct.py.
Plots need matplotlib (in requirements.txt); without it only the metrics, tables and report are logged.
"""

import re

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score,
                             roc_curve)

from skysat import schema
from training.train import predict_proba

MLP_ONLY = ("epochs", "batch_size", "lr", "weight_decay", "hidden", "dropout")  # ignored by the other models
IMPORTANCE_ROWS = 5000  # feature importance on a fixed sample of the validation rows: enough, and fast
# Passenger groups: the categories, plus bands of the numbers (edges are fixed, so runs are comparable)
BANDS = {
    "departure_delay_minutes": ([-1, 0, 15, 60, np.inf], ["on time", "1-15 min", "16-60 min", "> 60 min"]),
    "age": ([0, 25, 40, 60, np.inf], ["< 25", "25-39", "40-59", "60+"]),
    "flight_distance": ([0, 500, 1500, np.inf], ["< 500", "500-1499", "1500+"]),
}
BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f"


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


def passenger_groups(val_df: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """(column, value, row mask) for every category and every band of the numbers in BANDS."""
    out = [(col, value, (val_df[col] == value).to_numpy())
           for col, values in schema.CATEGORICAL_VALUES.items() for value in values]
    for col, (edges, labels) in BANDS.items():
        band = pd.cut(val_df[col], edges, labels=labels)
        out += [(col, label, (band == label).to_numpy()) for label in labels]
    return out


def metrics_by_group(val_df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Scores per passenger group: a model can be great on average and weak for one group."""
    y, pred = val_df[schema.TARGET].to_numpy(dtype=int), (p >= 0.5).astype(int)
    out = []
    for col, value, m in passenger_groups(val_df):
        if m.sum() == 0:
            continue
        auc = roc_auc_score(y[m], p[m]) if len(set(y[m])) == 2 else float("nan")
        out.append({"group": col, "value": value, "passengers": int(m.sum()),
                    "satisfied_share": round(float(y[m].mean()), 4),
                    "accuracy": round(float((pred[m] == y[m]).mean()), 4), "roc_auc": round(float(auc), 4)})
    return pd.DataFrame(out)


def _metric_name(*parts: str) -> str:
    """MLflow metric names allow letters, digits, _ - . / and spaces; `<` `>` `+` are spelled out."""
    s = ".".join(parts).replace("<", "under ").replace(">", "over ").replace("+", " plus")
    return re.sub(r"[^\w\-./ ]", "_", s)


def _report(bundle: dict, settings: dict, importance: pd.DataFrame, groups: pd.DataFrame) -> str:
    meta, metrics = bundle["meta"], bundle["meta"]["metrics"]
    big = groups[groups["passengers"] >= 200]  # tiny groups are too noisy to call "weakest"
    weakest = big.sort_values("accuracy").head(3)
    lines = [
        f"# Model report: {meta['params']['model']}", "",
        f"Trained on {meta['n_train_rows']:,} rows, scored on {meta['n_val_rows']:,} validation rows. "
        "Plots: `plots/overview.png` first.", "",
        "| Metric | Value |", "|---|---|",
        *[f"| {k} | {v:.4f} |" for k, v in metrics.items()], "",
        "## Strongest inputs (ROC AUC drop when shuffled)", "",
        *[f"{i + 1}. {r.feature}: {r.roc_auc_drop:.4f}" for i, r in importance.head(5).iterrows()], "",
        f"## Weakest passenger groups (overall accuracy {metrics['accuracy']:.4f})", "",
        *[f"- {r.group} = {r.value}: accuracy {r.accuracy:.4f}, ROC AUC {r.roc_auc:.4f} ({r.passengers:,} passengers)"
          for r in weakest.itertuples()],
        "", "All groups: `tables/metrics_by_group.csv`, `plots/groups.png`.", "",
        f"## Settings ({len(settings)})", "", *[f"- `{k}`: {v}" for k, v in settings.items()],
    ]
    return "\n".join(lines) + "\n"


def log_insights(bundle: dict, val_df: pd.DataFrame) -> None:
    """Inside an active MLflow run: loss curves, group metrics, plots, tables and report.md (module docstring)."""
    from mlflow.entities import Metric
    run_id = mlflow.active_run().info.run_id
    y = val_df[schema.TARGET].to_numpy(dtype=int)
    p = predict_proba(bundle, val_df)
    importance = feature_importance(bundle, val_df)
    groups = metrics_by_group(val_df, p)
    curves = loss_curves(bundle)

    now = int(pd.Timestamp.now().timestamp() * 1000)  # one request for all of them (DagsHub is slow per request)
    batch = [Metric(name, float(v), now, step) for name, loss in curves.items() for step, v in enumerate(loss, 1)]
    batch += [Metric(_metric_name("group", r.group, str(r.value), k), float(getattr(r, k)), now, 0)
              for r in groups.itertuples() for k in ("accuracy", "roc_auc") if not np.isnan(getattr(r, k))]
    client = mlflow.MlflowClient()
    for i in range(0, len(batch), 900):  # MLflow accepts up to 1000 metrics per batch
        client.log_batch(run_id, metrics=batch[i:i + 900])
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
    _plots(plt, matplotlib.ticker, y, p, importance, groups, curves, bundle["meta"]["params"]["model"])


# ------------------------------------------------------------------------------------------------- plots
# Each draw_* function draws one view on a given axis, so the same code makes the single plots and the overview.

def _caption(ax, text):
    ax.text(0, -0.2, text, transform=ax.transAxes, fontsize=7.5, color=GREY, va="top", wrap=True)


def draw_roc(ax, y, p):
    fpr, tpr, thr = roc_curve(y, p)
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"model (ROC AUC {roc_auc_score(y, p):.4f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="random guess (0.5)")
    i = np.argmin(np.abs(thr - 0.5))
    ax.plot(fpr[i], tpr[i], "o", color=RED, label=f"threshold 0.5: {tpr[i]:.1%} caught, {fpr[i]:.1%} false alarms")
    ax.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curve")
    ax.legend(loc="lower right", fontsize=7)
    zoom = ax.inset_axes([0.3, 0.32, 0.45, 0.4])  # the interesting part: the top-left corner
    zoom.plot(fpr, tpr, color=BLUE, lw=1.5)
    zoom.plot(fpr[i], tpr[i], "o", color=RED)
    zoom.set(xlim=(0, 0.15), ylim=(0.85, 1.0))
    zoom.set_title("zoom: top-left corner", fontsize=6)
    zoom.tick_params(labelsize=5)
    _caption(ax, "Closer to the top-left corner = better at telling satisfied from not satisfied.")


def draw_pr(ax, y, p):
    prec, rec, _ = precision_recall_curve(y, p)
    ax.plot(rec, prec, color=BLUE, lw=2, label=f"model (average precision {average_precision_score(y, p):.4f})")
    ax.axhline(y.mean(), ls="--", color=GREY, label=f"always 'satisfied' ({y.mean():.1%})")
    ax.set(xlabel="recall (share of satisfied found)", ylabel="precision (share of 'satisfied' answers right)",
           title="Precision vs. recall", ylim=(0, 1.02))
    ax.legend(loc="lower left", fontsize=7)
    _caption(ax, "Closer to the top-right = fewer mistakes for the same number of satisfied passengers found.")


def draw_threshold(ax, y, p):
    t = np.linspace(0.02, 0.98, 49)
    pred = p[None, :] >= t[:, None]
    tp = (pred & (y == 1)).sum(1)
    precision = tp / np.maximum(pred.sum(1), 1)
    recall = tp / max((y == 1).sum(), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    acc = (pred == (y == 1)).mean(1)
    for values, name, color in [(precision, "precision", BLUE), (recall, "recall", ORANGE), (f1, "F1", GREEN),
                                (acc, "accuracy", RED)]:
        ax.plot(t, values, color=color, lw=1.8, label=name)
    best = t[np.argmax(f1)]
    ax.axvline(0.5, color=GREY, ls="--", lw=1)
    ax.axvline(best, color=GREEN, ls=":", lw=1)
    ax.set(xlabel="decision threshold", ylabel="score", title=f"Scores by threshold (best F1 at {best:.2f})",
           ylim=(max(0, min(precision.min(), recall.min()) - 0.02), 1.0))
    ax.legend(loc="lower center", fontsize=7, ncol=2)
    _caption(ax, "Dashed: the threshold the API uses (0.5). A higher threshold = fewer, surer 'satisfied' answers.")


def draw_calibration(ax, y, p):
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    used = [b for b in range(10) if (idx == b).sum() >= 20]
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="perfectly honest")
    ax.plot([p[idx == b].mean() for b in used], [y[idx == b].mean() for b in used], "o-", color=BLUE, lw=2,
            label="model")
    ax.set(xlabel="predicted probability (bins of 0.1)", ylabel="real share satisfied", title="Calibration",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=7)
    ins = ax.inset_axes([0.58, 0.07, 0.4, 0.3])
    ins.hist([p[y == 0], p[y == 1]], bins=20, color=[BLUE, ORANGE], stacked=True)
    ins.set_yscale("log")
    ins.set_title("predictions (log count)", fontsize=6)
    ins.tick_params(labelsize=5)
    _caption(ax, "On the diagonal = when the model says 80%, about 80% really are satisfied.")


def draw_confusion(ax, y, p):
    cm = confusion_matrix(y, (p >= 0.5).astype(int))
    share = cm / cm.sum(1, keepdims=True)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    labels = ["not satisfied", "satisfied"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels, xlabel="model says",
           ylabel="passenger really was", title=f"Answers at threshold 0.5 (accuracy {np.trace(cm) / cm.sum():.1%})")
    for (i, j), n in np.ndenumerate(cm):
        ax.text(j, i, f"{n:,}\n{share[i, j]:.1%} of row", ha="center", va="center", fontsize=9,
                color="white" if share[i, j] > 0.5 else "black")
    _caption(ax, "Diagonal = right answers. Off-diagonal: missed satisfied (bottom-left), false 'satisfied' (top-right).")


def draw_importance(ax, importance, top=12):
    d = importance.head(top)[::-1]
    ax.barh(d["feature"], d["roc_auc_drop"], color=[RED if v < 0 else BLUE for v in d["roc_auc_drop"]])
    for i, v in enumerate(d["roc_auc_drop"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=7)
    ax.set(xlabel="ROC AUC lost when this input is shuffled", title=f"Most important inputs (top {top})")
    ax.tick_params(axis="y", labelsize=7)
    ax.margins(x=0.15)
    _caption(ax, "Long bar = the model relies on it. Near zero = the model would do as well without it.")


def draw_groups(ax, groups, overall):
    g = groups[groups["passengers"] >= 50].reset_index(drop=True)
    colors = [RED if a < overall - 0.01 else (GREEN if a > overall + 0.01 else BLUE) for a in g["accuracy"]]
    ax.barh(range(len(g)), g["accuracy"], color=colors)
    ax.axvline(overall, color="black", ls="--", lw=1, label=f"overall {overall:.1%}")
    ax.set_yticks(range(len(g)), [f"{r.group}: {r.value} ({r.passengers:,})" for r in g.itertuples()], fontsize=6.5)
    ax.invert_yaxis()
    ax.set(xlabel="accuracy", title="Accuracy per passenger group", xlim=(max(0.8, g["accuracy"].min() - 0.02), 1))
    ax.legend(loc="lower right", fontsize=7)
    _caption(ax, "Red: more than 1 point below overall (where the model is weakest). Group sizes in brackets.")


def draw_loss(ax, curves, ticker):
    for name, loss in curves.items():
        ax.plot(range(1, len(loss) + 1), loss, "o-", ms=3, label=name)
    ax.set(xlabel="epoch", ylabel="training loss", title="MLP training curve")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=7)
    _caption(ax, "Should fall and flatten. Still falling steeply = more epochs could help.")


def _plots(plt, ticker, y, p, importance, groups, curves, model_name):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                         "axes.titlesize": 10, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    overall = float(((p >= 0.5).astype(int) == y).mean())

    def save(fig, name):
        fig.tight_layout()
        mlflow.log_figure(fig, f"plots/{name}.png")
        plt.close(fig)

    singles = [("roc_curve", (5.5, 5.5), lambda ax: draw_roc(ax, y, p)),
               ("precision_recall", (5.5, 5.5), lambda ax: draw_pr(ax, y, p)),
               ("threshold", (6, 4.5), lambda ax: draw_threshold(ax, y, p)),
               ("calibration", (5.5, 5.5), lambda ax: draw_calibration(ax, y, p)),
               ("confusion_matrix", (5.5, 5), lambda ax: draw_confusion(ax, y, p)),
               ("feature_importance", (6.5, 5.5), lambda ax: draw_importance(ax, importance, top=15)),
               ("groups", (7, 7), lambda ax: draw_groups(ax, groups, overall))]
    if curves:
        singles.append(("train_loss", (6, 4), lambda ax: draw_loss(ax, curves, ticker)))
    for name, size, draw in singles:
        fig, ax = plt.subplots(figsize=size)
        draw(ax)
        save(fig, name)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5))
    draw_roc(axes[0, 0], y, p)
    draw_confusion(axes[0, 1], y, p)
    draw_calibration(axes[0, 2], y, p)
    draw_threshold(axes[1, 0], y, p)
    draw_importance(axes[1, 1], importance)
    draw_groups(axes[1, 2], groups, overall)
    fig.suptitle(f"{model_name}: validation ROC AUC {roc_auc_score(y, p):.4f}, accuracy {overall:.1%} "
                 f"on {len(y):,} passengers", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=4)
    mlflow.log_figure(fig, "plots/overview.png")
    plt.close(fig)

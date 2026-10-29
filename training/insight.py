"""Extra detail for an MLflow run, at no extra runs: the model's real settings, the training curve, and plots.

    settings = model_settings(bundle)        # {"model.n_estimators": 500, ...} for scikit-learn-style models
    log_insights(bundle, val_df)             # inside an active run: train_loss per epoch, ROC curve, confusion matrix

Called by training/ct.py. Plots need matplotlib (in requirements.txt); without it they are skipped with a note,
so a machine that only serves models never needs it.
"""

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from skysat import schema
from training.train import predict_proba

MLP_ONLY = ("epochs", "batch_size", "lr", "weight_decay", "hidden", "dropout")  # ignored by the other models


def run_params(params: dict) -> dict:
    """The training params that apply to this model type (the MLP-only ones are left out for the others)."""
    if params["model"] == "mlp":
        return params
    return {k: v for k, v in params.items() if k not in MLP_ONLY}


def model_settings(bundle: dict) -> dict:
    """The fitted estimator's own settings (trees, depth, ...), prefixed `model.`; empty for the MLP."""
    if "members" in bundle:  # ensemble: every member's settings, prefixed with its name
        return {f"{name}.{k}": v for name, member in bundle["members"].items()
                for k, v in (model_settings(member) or run_params(member["meta"]["params"])).items()}
    model = bundle["model"]
    if not hasattr(model, "get_params"):
        return {}
    simple = (str, int, float, bool, type(None))
    return {f"model.{k}": v if isinstance(v, simple) else str(v) for k, v in sorted(model.get_params().items())}


def log_insights(bundle: dict, val_df: pd.DataFrame) -> None:
    """Inside an active MLflow run: the MLP's loss per epoch, ROC curve and confusion matrix on validation."""
    run_id = mlflow.active_run().info.run_id
    loss = bundle["meta"].get("train_loss", [])
    if loss:  # one request for all epochs (DagsHub is slow per request)
        from mlflow.entities import Metric
        now = int(pd.Timestamp.now().timestamp() * 1000)
        mlflow.MlflowClient().log_batch(run_id, metrics=[Metric("train_loss", float(v), now, step)
                                                         for step, v in enumerate(loss, start=1)])
    try:
        import matplotlib
        matplotlib.use("Agg")  # no screen needed (CT runs on a server)
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: plots skipped")
        return

    y = val_df[schema.TARGET].to_numpy(dtype=int)
    p = predict_proba(bundle, val_df)
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC {roc_auc_score(y, p):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="random guess")
    ax.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curve (validation)")
    ax.legend(loc="lower right")
    mlflow.log_figure(fig, "plots/roc_curve.png")
    plt.close(fig)

    cm = confusion_matrix(y, (p >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    labels = ["not satisfied", "satisfied"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
           xlabel="predicted", ylabel="real answer", title="Confusion matrix (validation, threshold 0.5)")
    for (i, j), n in np.ndenumerate(cm):
        ax.text(j, i, f"{n:,}", ha="center", va="center", color="white" if n > cm.max() / 2 else "black")
    mlflow.log_figure(fig, "plots/confusion_matrix.png")
    plt.close(fig)

"""Step 3 – Train the model inside an MLflow run and register it.

    1. load data/processed/train.csv and validation.csv (from `python -m training.prepare_data`)
    2. run_training(train_df, val_df, params) -> (model.pt, metrics)
    3. log params, metrics, tags (git commit, purpose=training) and model.pt to MLflow
    4. quality gate: register the model only if roc_auc >= ROC_AUC_THRESHOLD
    5. move the `champion` alias only if the new version beats the current champion

Start MLflow first (`docker compose up mlflow`), then run from the project root:
    python -m training.track
    python -m training.track --stub              # fake training, no data needed
    python -m training.track --stub --stub-auc 0.94
"""

import argparse
import os
import subprocess

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from skysat import schema
from skysat.data import PROCESSED_DIR

EXPERIMENT = "skysat"
MODEL_NAME = "skysat-satisfaction"
ALIAS = "champion"
# Written down by the team: a model must reach this validation ROC AUC to be registered.
# The LightGBM baseline from Kaggle reaches ~0.958.
ROC_AUC_THRESHOLD = 0.95

def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = [PROCESSED_DIR / "train.csv", PROCESSED_DIR / "validation.csv"]
    for path in paths:
        if not path.exists():
            raise SystemExit(f"ERROR: {path} not found. Run `python -m training.prepare_data` first.")
    return pd.read_csv(paths[0]), pd.read_csv(paths[1])


def current_champion_auc(client: MlflowClient) -> float | None:
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except MlflowException:
        return None
    return client.get_run(champion.run_id).data.metrics.get("roc_auc")


def register(client: MlflowClient, run, roc_auc: float) -> None:
    if roc_auc < ROC_AUC_THRESHOLD:
        print(f"Not registered: roc_auc {roc_auc:.4f} < threshold {ROC_AUC_THRESHOLD}")
        return

    try:
        client.create_registered_model(MODEL_NAME)
    except MlflowException:
        pass  # already exists
    version = client.create_model_version(
        MODEL_NAME, source=f"{run.info.artifact_uri}/model", run_id=run.info.run_id
    )
    print(f"Registered {MODEL_NAME} version {version.version}")

    champion_auc = current_champion_auc(client)
    if champion_auc is None or roc_auc > champion_auc:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
        before = "none" if champion_auc is None else f"{champion_auc:.4f}"
        print(f"Version {version.version} is the new {ALIAS} (roc_auc {roc_auc:.4f}, previous {before})")
    else:
        print(f"{ALIAS} stays: its roc_auc {champion_auc:.4f} >= {roc_auc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train, log to MLflow and register the model")
    parser.add_argument("--stub", action="store_true", help="use the fake run_training (no data needed)")
    parser.add_argument("--stub-auc", type=float, help="roc_auc the fake training returns")
    args = parser.parse_args()

    if args.stub:
        from training.stub_train import run_training
        train_df = val_df = pd.DataFrame(columns=schema.FEATURES + [schema.TARGET])
        params = {} if args.stub_auc is None else {"stub_roc_auc": args.stub_auc}
    else:
        from training.train import DEFAULT_PARAMS, run_training
        train_df, val_df = load_data()
        params = dict(DEFAULT_PARAMS)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050"))
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    tags = {"git_commit": git_commit(), "purpose": "training", "stub": str(args.stub).lower()}
    with mlflow.start_run(tags=tags) as run:
        mlflow.log_params(params)
        mlflow.log_params({"train_rows": len(train_df), "validation_rows": len(val_df)})

        model_path, metrics = run_training(train_df, val_df, params)

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        print(f"Run {run.info.run_id}: " + ", ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

        register(client, run, metrics["roc_auc"])


if __name__ == "__main__":
    main()

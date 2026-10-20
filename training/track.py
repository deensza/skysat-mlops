"""Step 3 – Train the model inside an MLflow run and register it.

    1. load data/processed/train.csv and validation.csv (from `python -m training.prepare_data`)
    2. run_training(train_df, val_df, params) -> (model.pt, metrics)
    3. log params, metrics, tags (git commit, purpose=training) and model.pt to MLflow
    4. quality gate: register the model only if roc_auc >= ROC_AUC_THRESHOLD
    5. move the `champion` alias only if the new version beats the current champion, both scored on the
       same validation rows (champion_auc_on)

Start MLflow first (`docker compose up mlflow`), then run from the project root:
    python -m training.track
    python -m training.track --stub              # fake training, no data needed (local MLflow only)
    python -m training.track --stub --stub-auc 0.94
"""

import argparse
import os
import subprocess
import tempfile
from urllib.parse import urlparse

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import roc_auc_score

from skysat import schema
from skysat.data import PROCESSED_DIR

EXPERIMENT = "skysat"
MODEL_NAME = "skysat-satisfaction"
ALIAS = "champion"
# Written down by the team: a model must reach this validation ROC AUC to be registered. Real models on this
# dataset score 0.98-0.996 (README step 8), so the gate catches broken models, not merely weaker ones.
ROC_AUC_THRESHOLD = 0.95


def git_commit() -> str:
    if os.environ.get("GIT_COMMIT"):  # lets a build without a .git folder still say which commit it is
        return os.environ["GIT_COMMIT"]
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


def champion_bundle(client: MlflowClient) -> tuple[str, dict] | None:
    """The current champion as (version, bundle loaded the way the API loads it), or None if there is none yet."""
    from training.train import load_bundle, model_file  # imported here so `--stub` still works without PyTorch
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except MlflowException:
        return None
    folder = mlflow.artifacts.download_artifacts(artifact_uri=version.source, dst_path=tempfile.mkdtemp())
    return version.version, load_bundle(model_file(folder))


def champion_auc_on(client: MlflowClient, df: pd.DataFrame) -> float | None:
    """The champion's ROC AUC on these rows, so a challenger is compared on the same passengers.

    The roc_auc stored with the champion's run may come from other rows (drift retraining scores on recent
    passengers), so comparing against it would be comparing different exams. None if there is no champion.
    """
    from training.train import predict_proba
    champion = champion_bundle(client)
    if champion is None:
        return None
    version, bundle = champion
    auc = float(roc_auc_score(df[schema.TARGET], predict_proba(bundle, df)))
    print(f"Champion (version {version}) on the same rows: roc_auc {auc:.4f}")
    return auc


def current_champion_auc(client: MlflowClient) -> float | None:
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except MlflowException:
        return None
    return client.get_run(champion.run_id).data.metrics.get("roc_auc")


def register(client: MlflowClient, run, roc_auc: float, champion_auc: float | None = None) -> str | None:
    """Quality gate. champion_auc: the champion's score on the same rows as roc_auc (champion_auc_on);
    if not given (only `--stub`, which has no real rows), the roc_auc stored with the champion's run is used.
    Returns the new version number, or None if the gate rejected the model."""
    if roc_auc < ROC_AUC_THRESHOLD:
        print(f"Not registered: roc_auc {roc_auc:.4f} < threshold {ROC_AUC_THRESHOLD}")
        return None

    try:
        client.create_registered_model(MODEL_NAME)
    except MlflowException:
        pass  # already exists
    version = client.create_model_version(
        MODEL_NAME, source=f"{run.info.artifact_uri}/model", run_id=run.info.run_id
    )
    print(f"Registered {MODEL_NAME} version {version.version}")

    if champion_auc is None:
        champion_auc = current_champion_auc(client)
    if champion_auc is None or roc_auc > champion_auc:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
        before = "none" if champion_auc is None else f"{champion_auc:.4f}"
        print(f"Version {version.version} is the new {ALIAS} (roc_auc {roc_auc:.4f}, previous {before})")
    else:
        print(f"{ALIAS} stays: its roc_auc {champion_auc:.4f} >= {roc_auc:.4f}")
    return version.version


def main():
    parser = argparse.ArgumentParser(description="Train, log to MLflow and register the model")
    parser.add_argument("--stub", action="store_true", help="use the fake run_training (no data needed)")
    parser.add_argument("--stub-auc", type=float, help="roc_auc the fake training returns")
    args = parser.parse_args()

    uri = os.environ.get("MLFLOW_TRACKING_URI") or "http://localhost:5050"
    if args.stub and urlparse(uri).hostname not in ("localhost", "127.0.0.1"):
        # a stub "model" is a dummy file: as champion it would break every API that loads it, the team's too
        raise SystemExit(f"ERROR: --stub only against a local MLflow, not {uri}")
    if args.stub:
        from training.stub_train import run_training
        train_df = val_df = pd.DataFrame(columns=schema.FEATURES + [schema.TARGET])
        params = {} if args.stub_auc is None else {"stub_roc_auc": args.stub_auc}
    else:
        from training.train import DEFAULT_PARAMS, run_training
        train_df, val_df = load_data()
        params = dict(DEFAULT_PARAMS)

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    champion_auc = None if args.stub else champion_auc_on(client, val_df)

    tags = {"git_commit": git_commit(), "purpose": "training", "stub": str(args.stub).lower()}
    with mlflow.start_run(tags=tags) as run:
        mlflow.log_params(params)
        mlflow.log_params({"train_rows": len(train_df), "validation_rows": len(val_df)})

        model_path, metrics = run_training(train_df, val_df, params)

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        print(f"Run {run.info.run_id}: " + ", ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

        register(client, run, metrics["roc_auc"], champion_auc=champion_auc)


if __name__ == "__main__":
    main()

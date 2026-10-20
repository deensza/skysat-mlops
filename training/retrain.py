"""Step 7 – Drift detection and retraining.

    1. read the latest --window predictions from PostgreSQL (inputs + real answers from /feedback)
    2. PSI of every feature: recent traffic vs the training data (skysat/monitoring.py)
    3. no feature with PSI >= 0.2 -> no drift, stop
    4. drift -> split the labelled recent passengers 70/30 (stratified):
         challenger = run_training(train.csv + 70% recent x --recent-weight, 30% recent holdout)
         champion   = the model serving now, scored on the SAME 30% holdout
    5. log everything to MLflow (purpose=retraining) and apply the quality gate from track.py:
       registered if roc_auc >= 0.95, becomes `champion` only if it beats the champion on the holdout.
       The API copies pick up the new champion by themselves within ~10 s.

The recent passengers are repeated --recent-weight times (default 5) so ~1,400 new rows are not
drowned by 83,000 old ones. Chosen in an offline test on unseen storm traffic: weight 1, 5, 10, 20
gave AUC 0.977, 0.986, 0.986, 0.985 (champion 0.954), all keeping AUC ~0.994 on normal passengers.

Run from the project root (needs `docker compose up` and the PyTorch packages):
    python -m training.retrain
    python -m training.retrain --check-only    # only measure drift
"""

import argparse
import os
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import psycopg
from mlflow import MlflowClient
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from skysat import schema
from skysat.monitoring import PSI_THRESHOLD, feature_psi
from training.track import ALIAS, EXPERIMENT, MODEL_NAME, git_commit, load_data, register
from training.train import DEFAULT_PARAMS, load_bundle, model_file, predict_proba, run_training

SEED = 42


def recent_predictions(window: int) -> pd.DataFrame:
    """The latest `window` predictions: the 22 inputs and the real answer (NaN if not received)."""
    url = os.environ.get("DATABASE_URL", "postgresql://skysat:skysat@localhost:5432/skysat")
    with psycopg.connect(url) as conn:
        rows = conn.execute(
            "SELECT features, actual FROM predictions ORDER BY created_at DESC LIMIT %s", (window,)
        ).fetchall()
    if not rows:
        raise SystemExit("ERROR: no predictions in the database yet. Send traffic with the simulator first.")
    df = pd.DataFrame([features for features, _ in rows])[schema.FEATURES]
    df[schema.TARGET] = [actual for _, actual in rows]
    return df


def champion_bundle(client: MlflowClient):
    version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    folder = mlflow.artifacts.download_artifacts(artifact_uri=version.source, dst_path=tempfile.mkdtemp())
    return version.version, load_bundle(model_file(folder))


def main():
    parser = argparse.ArgumentParser(description="Check drift; retrain and promote if needed")
    parser.add_argument("--window", type=int, default=2000, help="latest predictions to check (default 2000)")
    parser.add_argument("--recent-weight", type=int, default=5, help="times the recent rows are repeated")
    parser.add_argument("--min-labelled", type=int, default=500, help="labelled rows needed to retrain")
    parser.add_argument("--check-only", action="store_true", help="measure drift, do not retrain")
    args = parser.parse_args()

    train_df, val_df = load_data()
    recent = recent_predictions(args.window)

    scores = feature_psi(train_df, recent)
    print(f"PSI, latest {len(recent)} predictions vs training data (drift if >= {PSI_THRESHOLD}):")
    for col, value in scores.items():
        print(f"  {col:<36}{value:7.3f}{'  <- drift' if value >= PSI_THRESHOLD else ''}")
    drifted = list(scores[scores >= PSI_THRESHOLD].index)
    if not drifted:
        print("No drift: the champion keeps serving.")
        return
    print(f"Drift in {len(drifted)} feature(s): {', '.join(drifted)}")
    if args.check_only:
        return

    labelled = recent.dropna(subset=[schema.TARGET]).astype({schema.TARGET: int})
    if len(labelled) < args.min_labelled:
        raise SystemExit(f"Only {len(labelled)} predictions have a real answer; need {args.min_labelled} to retrain.")
    recent_train, holdout = train_test_split(
        labelled, test_size=0.3, stratify=labelled[schema.TARGET], random_state=SEED
    )

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or "http://localhost:5050")
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    champion_version, champion = champion_bundle(client)
    champion_auc = float(roc_auc_score(holdout[schema.TARGET], predict_proba(champion, holdout)))
    print(f"Champion (version {champion_version}) on the recent holdout: roc_auc {champion_auc:.4f}")

    combined = pd.concat([train_df] + [recent_train] * args.recent_weight, ignore_index=True)
    tags = {"git_commit": git_commit(), "purpose": "retraining", "replaces_version": champion_version}
    with mlflow.start_run(tags=tags) as run:
        params = {**DEFAULT_PARAMS, **champion["meta"]["params"]}  # same recipe (model type) as the champion
        mlflow.log_params(params)
        mlflow.log_params({
            "window": args.window, "recent_weight": args.recent_weight, "psi_threshold": PSI_THRESHOLD,
            "train_rows": len(train_df), "recent_train_rows": len(recent_train), "holdout_rows": len(holdout),
        })
        mlflow.log_metrics({f"psi_{col}": value for col, value in scores.items()})

        with tempfile.TemporaryDirectory() as out:
            model_path, metrics = run_training(combined, holdout, params, Path(out))
            mlflow.log_artifact(str(model_path), artifact_path="model")
            # how the challenger does on the original validation passengers (no storm), for the record
            challenger = load_bundle(model_path)
            metrics["normal_validation_roc_auc"] = float(
                roc_auc_score(val_df[schema.TARGET], predict_proba(challenger, val_df)))
        metrics["champion_roc_auc"] = champion_auc
        mlflow.log_metrics(metrics)
        print(f"Challenger on the recent holdout: roc_auc {metrics['roc_auc']:.4f}, "
              f"accuracy {metrics['accuracy']:.4f} (normal validation roc_auc "
              f"{metrics['normal_validation_roc_auc']:.4f})")

        register(client, run, metrics["roc_auc"], champion_auc=champion_auc)


if __name__ == "__main__":
    main()

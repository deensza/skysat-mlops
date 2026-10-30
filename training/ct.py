"""Step 8 – Continuous training (CT): train one or more models, register the good ones, promote the best.

Normally runs on GitHub's servers (.github/workflows/ct.yml), so it works while every team PC is off.
Results go to the team's shared MLflow on DagsHub, so everyone sees the same runs, versions and champion.

    make add-ct ARGS="--model random_forest"      # ask GitHub to train it (waits in GitHub's run queue)
    make add-ct ARGS="--model mlp --rating-one-hot --epochs 30 --name mlp-30-epochs"
    make add-ct ARGS=--all                        # the 9 standard candidates
    make list-ct                                  # recent CT runs on GitHub
    make ct-local ARGS="--model mlp --epochs 5"   # the same training on this computer (needs `make data`)

For every model:
    1. train on data/processed/train.csv, measure on validation.csv (training.train.run_training)
    2. log an MLflow run (tags purpose=ct, model_type) with params, metrics and the model file
    3. quality gate from track.py: registered if roc_auc >= 0.95, `champion` only if it beats the current one;
       the API copies switch to a new champion within ~10 s
Exits with an error if no model passed the gate, so the CT run on GitHub turns red.
"""

import argparse
import os
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from skysat import schema
from training.data_card import ORIGINAL_URL, data_tags
from training.insight import log_insights, model_settings, run_params
from training.track import EXPERIMENT, ROC_AUC_THRESHOLD, champion_auc_on, git_commit, load_data, register
from training.train import DEFAULT_PARAMS, SKLEARN_MODELS, load_bundle, run_training

MODELS = ["mlp", "ensemble", *SKLEARN_MODELS]

# The 9 standard candidates (--all). Linear models cannot learn the step-shaped ratings, so logistic
# regression gets rating_one_hot; the trees split on the raw 0-5 values by themselves.
# extra_trees, decision_tree and knn train only with --model (they never beat these).
CANDIDATES = {
    "logistic_regression": {"model": "logistic_regression", "rating_one_hot": True},
    "random_forest": {"model": "random_forest"},
    "gradient_boosting": {"model": "gradient_boosting"},
    "gradient_boosting_tuned": {"model": "gradient_boosting_tuned"},
    "lightgbm": {"model": "lightgbm"},
    "xgboost": {"model": "xgboost"},
    "mlp": {"model": "mlp"},
    "mlp_rating_one_hot": {"model": "mlp", "rating_one_hot": True},
    "ensemble": {"model": "ensemble"},
}


def parse_jobs(argv: list[str] | None = None) -> dict:
    """Command line -> {job name: overrides of DEFAULT_PARAMS}."""
    parser = argparse.ArgumentParser(description="Train models, register the good ones, promote the best")
    parser.add_argument("--all", action="store_true", help="the 9 standard candidates")
    parser.add_argument("--model", choices=MODELS, help="one model type")
    parser.add_argument("--name", help="label for the run (default: the model name)")
    parser.add_argument("--rating-one-hot", action="store_true", help="add one-hot columns for every 0-5 rating")
    parser.add_argument("--drop-features", nargs="*", metavar="NAME", help="schema feature names to leave out")
    parser.add_argument("--epochs", type=int, help="MLP only")
    parser.add_argument("--lr", type=float, help="MLP only")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)

    if args.all == bool(args.model):
        parser.error("give either --all or --model")
    if unknown := set(args.drop_features or []) - set(schema.FEATURES):
        parser.error(f"unknown --drop-features {sorted(unknown)}; use names from skysat/schema.py")
    if args.all:
        return CANDIDATES
    params = {"model": args.model, "rating_one_hot": args.rating_one_hot}
    optional = {"drop_features": args.drop_features, "epochs": args.epochs, "lr": args.lr, "seed": args.seed}
    params.update({k: v for k, v in optional.items() if v is not None})
    return {args.name or args.model + ("_rating_one_hot" if args.rating_one_hot else ""): params}


def train_and_register(client: MlflowClient, train_df, val_df, name: str, overrides: dict,
                       champion_auc: float | None) -> tuple[float, str | None]:
    """Train one model, log it to MLflow, apply the quality gate. champion_auc: the champion's score on the same
    validation rows (None: no champion yet). Returns (roc_auc, registered version or None)."""
    params = {**DEFAULT_PARAMS, **overrides}
    tags = {"git_commit": git_commit(), "purpose": "ct", "model_type": params["model"], **data_tags()}
    with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(run_name=f"ct-{name}", tags=tags) as run:
        # which data: schema, row count and digest (no rows), the same entries as the data card (make data-card)
        for df, context in ((train_df, "training"), (val_df, "validation")):
            mlflow.log_input(mlflow.data.from_pandas(df, source=ORIGINAL_URL, name=f"processed/{context}",
                                                     targets=schema.TARGET), context=context)
        mlflow.log_params(run_params(params))  # only the settings this model type uses
        mlflow.log_params({"train_rows": len(train_df), "validation_rows": len(val_df)})
        path, metrics = run_training(train_df, val_df, params, Path(tmp))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(path, artifact_path="model")
        bundle = load_bundle(path)
        mlflow.log_params(model_settings(bundle))  # the fitted model's own settings (trees, depth, ...)
        log_insights(bundle, val_df)  # training curve (MLP), ROC curve, confusion matrix
        # MLflow 3 "logged model": the experiment's Models tab lists it with its settings and scores, linked to the
        # run and to the registry version. Same model file, so the API loads it as before.
        settings = {k: str(v) for k, v in {**run_params(params), **model_settings(bundle)}.items()}
        logged = mlflow.initialize_logged_model(name=f"ct-{name}", source_run_id=run.info.run_id,
                                                model_type=params["model"], params=settings)
        client.log_model_artifacts(logged.model_id, str(Path(path).parent))
        mlflow.log_metrics(metrics, model_id=logged.model_id)
        mlflow.finalize_logged_model(logged.model_id, "READY")
        version = register(client, run, metrics["roc_auc"], champion_auc=champion_auc, model_id=logged.model_id)
    return metrics["roc_auc"], version


def main():
    jobs = parse_jobs()
    train_df, val_df = load_data()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or "http://localhost:5050")
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    print(f"MLflow: {mlflow.get_tracking_uri()}")

    champion_auc = champion_auc_on(client, val_df)  # the champion takes the same exam as the candidates
    results = {}
    for name, overrides in jobs.items():
        print(f"\n== {name} {overrides} ==", flush=True)
        auc, version = results[name] = train_and_register(client, train_df, val_df, name, overrides, champion_auc)
        if version and (champion_auc is None or auc > champion_auc):
            champion_auc = auc  # register() just made this one the champion

    print(f"\nCT results (quality gate: roc_auc >= {ROC_AUC_THRESHOLD}):")
    for name, (auc, version) in sorted(results.items(), key=lambda r: -r[1][0]):
        print(f"  {name:<22} roc_auc {auc:.4f}  "
              f"{f'registered as version {version}' if version else 'rejected by the gate'}")
    if not any(version for _, version in results.values()):
        raise SystemExit("ERROR: no model passed the quality gate; the champion keeps serving.")


if __name__ == "__main__":
    main()

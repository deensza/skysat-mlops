"""Step 7 – Drift detection and retraining (on GitHub every hour: .github/workflows/drift.yml).

    1. read the latest --window predictions the champion made (inputs + real answers from /feedback):
       from PostgreSQL, or with --source from the server's API (GET /predictions/recent, behind its login)
    2. PSI of every feature: that traffic vs the champion's drift reference (skysat/monitoring.py), which is
       the traffic the champion was last checked on: drift_reference.json of the newest retraining run that
       challenged it, else of its own training run; the original training data if neither has one
    3. no feature with PSI >= 0.2 -> no drift, stop
    4. drift -> split the labelled recent passengers 70/30 (stratified):
         challenger = run_training(train.csv + 70% recent x --recent-weight, 30% recent holdout)
         champion   = the model serving now, scored on the SAME 30% holdout
    5. log everything to MLflow (purpose=retraining) and apply the quality gate from track.py: registered if
       roc_auc >= 0.95, becomes `champion` only if it beats the champion on the holdout. The APIs switch to it by
       themselves. If it was registered (a fair comparison), the recent traffic's profile (shares per bin, no rows)
       becomes the champion's new drift reference. Below the gate: error (a red run on GitHub), nothing moves,
       and the next check tries again with newer passengers.

Why the reference moves: compared with the original data, storm traffic stays "drift" after the model has
adapted to it, and the check would retrain every hour. Compared with the last traffic the champion was checked
on, the storm counts once; when it ends, normal traffic counts as drift again and the model adapts back.

The recent passengers are repeated --recent-weight times (default 5) so ~1,400 new rows are not
drowned by 83,000 old ones. Chosen in an offline test on unseen storm traffic: weight 1, 5, 10, 20
gave AUC 0.977, 0.986, 0.986, 0.985 (champion 0.954), all keeping AUC ~0.994 on normal passengers.

Run from the project root (needs the PyTorch packages; and `docker compose up` without --source):
    python -m training.retrain
    python -m training.retrain --check-only                         # only measure drift
    SKYSAT_LOGIN=user:password python -m training.retrain --source https://<server>
"""

import argparse
import base64
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from skysat import schema
from skysat.monitoring import PSI_THRESHOLD, REFERENCE_FILE, profile, profile_psi
from training.track import (ALIAS, EXPERIMENT, MODEL_NAME, ROC_AUC_THRESHOLD, champion_bundle, git_commit, load_data,
                            register)

SEED = 42


def to_frame(rows: list[tuple[dict, int | None]]) -> pd.DataFrame:
    """(features, actual) pairs -> the 22 inputs and the real answer (NaN if not received)."""
    df = pd.DataFrame([features for features, _ in rows], columns=schema.FEATURES)
    df[schema.TARGET] = pd.array([actual for _, actual in rows], dtype="Int64").astype("float")
    return df


def recent_from_database(window: int, version: str) -> pd.DataFrame:
    import psycopg  # only needed here: the GitHub check reads through the API instead
    url = os.environ.get("DATABASE_URL", "postgresql://skysat:skysat@localhost:5432/skysat")
    with psycopg.connect(url) as conn:
        rows = conn.execute(
            "SELECT features, actual FROM predictions WHERE model_version = %s ORDER BY created_at DESC LIMIT %s",
            (version, window),
        ).fetchall()
    return to_frame(rows)


def recent_from_server(source: str, window: int, version: str) -> pd.DataFrame:
    """GET /predictions/recent through the gateway. SKYSAT_LOGIN=user:password for its login."""
    url = f"{source.rstrip('/')}/predictions/recent?limit={window}&model_version={version}"
    request = urllib.request.Request(url)
    if login := os.environ.get("SKYSAT_LOGIN"):
        request.add_header("Authorization", "Basic " + base64.b64encode(login.encode()).decode())
    try:
        with urllib.request.urlopen(request, timeout=120) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as err:
        hint = " (check SKYSAT_LOGIN)" if err.code == 401 else ""
        raise SystemExit(f"ERROR: {source} answered {err.code} {err.reason}{hint}") from None
    return to_frame([(row["features"], row["actual"]) for row in body["rows"]])


def drift_reference(client: MlflowClient, version: str, train_df: pd.DataFrame) -> tuple[dict, str]:
    """The profile the champion is compared with, and where it comes from (see the module docstring)."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    challengers = client.search_runs(
        [experiment.experiment_id], max_results=20, order_by=["attributes.start_time DESC"],
        filter_string=f"tags.replaces_version = '{version}' and attributes.status = 'FINISHED'",
    ) if experiment else []
    places = [(run.info.run_id, f"the traffic of the last retraining that challenged version {version}")
              for run in challengers]
    places.append((client.get_model_version(MODEL_NAME, version).run_id, f"the data version {version} was trained on"))
    for run_id, where in places:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=REFERENCE_FILE, dst_path=tmp)
                return json.loads(Path(path).read_text()), where
        except Exception:  # older runs have no drift reference (the error type depends on the artifact store)
            continue
    return profile(train_df), f"the original training data (version {version}'s run has no {REFERENCE_FILE})"


def report(lines: list[str]) -> None:
    """Print, and on GitHub also put it on the run's summary page (numbers only, never rows)."""
    print("\n".join(lines))
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as f:
            f.write("\n".join(["```", *lines, "```", ""]))


def say_drift(drift: bool) -> None:
    """On GitHub, a later step can read it: steps.<id>.outputs.drift."""
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as f:
            f.write(f"drift={str(drift).lower()}\n")


def retrain(client: MlflowClient, train_df, val_df, recent, scores, champion_version: str, args) -> None:
    from training.train import DEFAULT_PARAMS, load_bundle, predict_proba, run_training  # PyTorch

    labelled = recent.dropna(subset=[schema.TARGET]).astype({schema.TARGET: int})
    if len(labelled) < args.min_labelled:
        raise SystemExit(f"Only {len(labelled)} predictions have a real answer; need {args.min_labelled} to retrain.")
    recent_train, holdout = train_test_split(
        labelled, test_size=0.3, stratify=labelled[schema.TARGET], random_state=SEED
    )

    _, champion = champion_bundle(client)
    champion_auc = float(roc_auc_score(holdout[schema.TARGET], predict_proba(champion, holdout)))
    print(f"Champion (version {champion_version}) on the recent holdout: roc_auc {champion_auc:.4f}")

    combined = pd.concat([train_df] + [recent_train] * args.recent_weight, ignore_index=True)
    tags = {"git_commit": git_commit(), "purpose": "retraining", "replaces_version": champion_version,
            "source": "server" if args.source else "database"}
    with mlflow.start_run(tags=tags) as run:
        params = {**DEFAULT_PARAMS, **champion["meta"]["params"]}  # same recipe (model type) as the champion
        mlflow.log_params(params)
        mlflow.log_params({
            "window": args.window, "recent_weight": args.recent_weight, "psi_threshold": PSI_THRESHOLD,
            "train_rows": len(train_df), "recent_rows": len(recent), "recent_train_rows": len(recent_train),
            "holdout_rows": len(holdout),
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

        version = register(client, run, metrics["roc_auc"], champion_auc=champion_auc)
        if version:
            # a fair comparison on this traffic happened, so it becomes the champion's next reference, whichever
            # model won (no rows). A challenger below the gate moves nothing: the next check tries again.
            mlflow.log_dict(profile(recent), REFERENCE_FILE)
    won = version and metrics["roc_auc"] > champion_auc
    report([f"Retrained: challenger roc_auc {metrics['roc_auc']:.4f} vs champion {champion_auc:.4f} "
            f"on {len(holdout)} recent passengers -> "
            + (f"version {version} is the new champion" if won else f"version {champion_version} keeps serving")])
    if not version:  # red run on GitHub: the traffic changed and no model good enough came out, a person should look
        raise SystemExit(f"ERROR: the challenger did not pass the quality gate (roc_auc >= {ROC_AUC_THRESHOLD}).")


def main():
    parser = argparse.ArgumentParser(description="Check drift; retrain and promote if needed")
    parser.add_argument("--source", help="the API's base URL (e.g. https://<server>); default: the database")
    parser.add_argument("--window", type=int, default=2000, help="latest predictions to check (default 2000)")
    parser.add_argument("--min-rows", type=int, default=500, help="predictions needed to judge drift")
    parser.add_argument("--recent-weight", type=int, default=5, help="times the recent rows are repeated")
    parser.add_argument("--min-labelled", type=int, default=500, help="labelled rows needed to retrain")
    parser.add_argument("--check-only", action="store_true", help="measure drift, do not retrain")
    args = parser.parse_args()

    train_df, val_df = load_data()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or "http://localhost:5050")
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    print(f"MLflow: {mlflow.get_tracking_uri()}")
    try:
        champion_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version
    except MlflowException:
        raise SystemExit("ERROR: there is no champion yet. Train one first (make ct-local).") from None

    if args.source:
        recent = recent_from_server(args.source, args.window, champion_version)
    else:
        recent = recent_from_database(args.window, champion_version)
    if len(recent) < args.min_rows:
        report([f"Version {champion_version} has made {len(recent)} predictions so far; "
                f"{args.min_rows} are needed to judge drift. Nothing to do yet."])
        say_drift(False)
        return

    reference, where = drift_reference(client, champion_version, train_df)
    scores = profile_psi(reference, recent)
    drifted = list(scores[scores >= PSI_THRESHOLD].index)
    lines = [f"PSI, latest {len(recent)} predictions of version {champion_version} vs {where} "
             f"({reference['rows']} rows); drift if >= {PSI_THRESHOLD}:"]
    lines += [f"  {col:<36}{value:7.3f}{'  <- drift' if value >= PSI_THRESHOLD else ''}"
              for col, value in scores.items()]
    lines.append(f"Drift in {len(drifted)} feature(s): {', '.join(drifted)}" if drifted
                 else "No drift: the champion keeps serving.")
    report(lines)
    say_drift(bool(drifted))
    if drifted and not args.check_only:
        retrain(client, train_df, val_df, recent, scores, champion_version, args)


if __name__ == "__main__":
    main()

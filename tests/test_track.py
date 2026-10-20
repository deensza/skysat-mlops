"""Checks for the MLflow side (training/track.py) with a throw-away MLflow in a temporary folder."""

import mlflow
import pytest
from mlflow import MlflowClient

pytest.importorskip("torch")

from skysat import schema
from skysat.data import clean
from tests.test_data_and_drift import make_raw
from training import track
from training.train import run_training


@pytest.fixture
def local_mlflow(tmp_path):
    before = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    experiment = mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    yield MlflowClient(), experiment
    mlflow.set_tracking_uri(before)


def test_champion_is_scored_on_the_same_rows_not_by_its_stored_number(local_mlflow, tmp_path):
    client, experiment = local_mlflow
    df = clean(make_raw(n=900, seed=7))
    df[schema.TARGET] = ((df["online_boarding"] + df["seat_comfort"]) >= 6).astype(int)
    train, val = df.iloc[:700], df.iloc[700:]
    path, metrics = run_training(train, val, {"model": "logistic_regression", "rating_one_hot": True}, tmp_path)
    with mlflow.start_run(experiment_id=experiment) as run:
        mlflow.log_metric("roc_auc", 0.5)  # a score from other rows, like drift retraining's recent holdout
        mlflow.log_artifact(path, artifact_path="model")
    assert str(track.register(client, run, 0.99)) == "1"  # first model: registered and champion

    assert track.current_champion_auc(client) == 0.5  # the stored number: a different exam
    assert track.champion_auc_on(client, val) == pytest.approx(metrics["roc_auc"])  # the same exam


def test_stub_refuses_a_shared_mlflow(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dagshub.com/team/skysat-mlops.mlflow")
    monkeypatch.setattr("sys.argv", ["track", "--stub"])
    with pytest.raises(SystemExit, match="only against a local MLflow"):
        track.main()

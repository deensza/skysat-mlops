"""The automatic loop (steps 7-8) without a server or a real model: where the drift reference comes from, how
the server's answer becomes a table, and the weather calendar of the hourly traffic."""

import json
from datetime import datetime, timezone

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from simulator.drift import apply_storm
from simulator.simulate import scheduled_mode
from skysat import schema
from skysat.data import clean
from skysat.monitoring import PSI_THRESHOLD, REFERENCE_FILE, feature_psi, profile, profile_psi
from tests.test_data_and_drift import make_raw
from training import retrain


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A throw-away MLflow with one registered version; returns (client, experiment id, run id of version 1)."""
    before = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    experiment = mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    monkeypatch.setattr(retrain, "EXPERIMENT", "test")
    client = MlflowClient()
    with mlflow.start_run(experiment_id=experiment) as run:
        pass
    client.create_registered_model(retrain.MODEL_NAME)
    client.create_model_version(retrain.MODEL_NAME, source=run.info.artifact_uri, run_id=run.info.run_id)
    yield client, experiment, run.info.run_id
    mlflow.set_tracking_uri(before)


@pytest.fixture(scope="module")
def traffic():
    normal = clean(make_raw(n=3000, seed=1))
    return normal, apply_storm(clean(make_raw(n=1000, seed=2)), share=0.6, seed=0)


def test_reference_falls_back_to_the_training_data(registry, traffic):
    client, _, _ = registry
    normal, _ = traffic
    reference, where = retrain.drift_reference(client, "1", normal)
    assert "original training data" in where
    assert reference == profile(normal)


def test_reference_is_the_one_logged_with_the_champion(registry, traffic):
    client, _, run_id = registry
    normal, storm = traffic
    with mlflow.start_run(run_id=run_id):
        mlflow.log_dict(profile(storm), REFERENCE_FILE)
    reference, where = retrain.drift_reference(client, "1", normal)
    assert "version 1 was trained on" in where
    assert reference == json.loads(json.dumps(profile(storm)))


def test_a_later_challenger_moves_the_reference_even_if_it_lost(registry, traffic):
    client, experiment, run_id = registry
    normal, storm = traffic
    with mlflow.start_run(run_id=run_id):
        mlflow.log_dict(profile(normal), REFERENCE_FILE)
    with mlflow.start_run(experiment_id=experiment, tags={"replaces_version": "1"}):
        mlflow.log_dict(profile(storm), REFERENCE_FILE)  # checked on storm traffic; version 1 kept serving
    reference, where = retrain.drift_reference(client, "1", normal)
    assert "last retraining" in where
    # so storm traffic no longer counts as drift for version 1, while it still would against the training data
    assert profile_psi(reference, storm).max() < PSI_THRESHOLD <= feature_psi(normal, storm).max()


def test_a_profile_gives_the_same_psi_as_the_rows_even_after_json(traffic):
    normal, storm = traffic
    stored = json.loads(json.dumps(profile(normal)))  # what MLflow keeps: no rows, only bins and shares
    assert stored["rows"] == len(normal)
    pd.testing.assert_series_equal(profile_psi(stored, storm), feature_psi(normal, storm))


def test_server_rows_become_a_table_with_missing_answers_as_nan(traffic):
    normal, _ = traffic
    rows = [(normal.iloc[0][schema.FEATURES].to_dict(), 1), (normal.iloc[1][schema.FEATURES].to_dict(), None)]
    df = retrain.to_frame(rows)
    assert list(df.columns) == [*schema.FEATURES, schema.TARGET]
    assert df[schema.TARGET].iloc[0] == 1 and pd.isna(df[schema.TARGET].iloc[1])
    assert retrain.to_frame([]).empty


@pytest.mark.parametrize("when, mode", [
    ("2026-11-19 10:00", "drift"),   # demo day, inside the storm hours
    ("2026-11-19 05:59", "normal"),  # before the storm
    ("2026-11-19 18:00", "normal"),  # after it
    ("2026-11-18 10:00", "normal"),  # a day without a storm
])
def test_weather_calendar(when, mode):
    assert scheduled_mode(datetime.fromisoformat(when).replace(tzinfo=timezone.utc)) == mode

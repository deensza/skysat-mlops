"""A new champion serves only after it has answered a test passenger. Nothing here needs MLflow or a real model."""

from types import SimpleNamespace

import numpy as np
import pytest

from api import main
from skysat import schema


@pytest.fixture
def new_champion(monkeypatch, tmp_path):
    """MLflow says the champion is version 2; "downloading" it gives an empty folder and an empty bundle."""
    version = SimpleNamespace(version="2", run_id="run-2", source="unused")
    monkeypatch.setattr(main, "MlflowClient", lambda: SimpleNamespace(get_model_version_by_alias=lambda *_: version))
    monkeypatch.setattr(main.mlflow.artifacts, "download_artifacts", lambda **_: str(tmp_path))
    monkeypatch.setattr(main, "model_file", lambda folder: folder)
    monkeypatch.setattr(main, "load_bundle", lambda path: {})


def cannot_run(bundle, df):
    raise KeyError("mean")  # what the old image did with the ensemble champion


def test_model_this_code_cannot_run_is_refused_on_a_fresh_start(new_champion, monkeypatch):
    monkeypatch.setattr(main, "predict_proba", cannot_run)
    serving = main.ServingModel()
    with pytest.raises(KeyError):
        serving.refresh()
    assert serving.current is None  # so /health answers 503, not a false "ok"


def test_model_this_code_cannot_run_leaves_the_old_model_serving(new_champion, monkeypatch):
    monkeypatch.setattr(main, "predict_proba", cannot_run)
    serving = main.ServingModel()
    serving.current = ("1", "run-1", {}, "earlier")
    with pytest.raises(KeyError):
        serving.refresh()
    assert serving.current[0] == "1"


def test_model_that_answers_goes_live(new_champion, monkeypatch):
    seen = []

    def answers(bundle, df):
        seen.append(df)
        return np.array([0.7])

    monkeypatch.setattr(main, "predict_proba", answers)
    serving = main.ServingModel()
    serving.refresh()
    assert serving.current[0] == "2"
    assert list(seen[0].columns) == [schema.ID_COLUMN, *schema.FEATURES]  # same columns as a real request


def test_recent_predictions_need_the_prediction_log(monkeypatch):
    monkeypatch.setattr(main, "db", None)
    with pytest.raises(main.HTTPException) as err:
        main.recent_predictions(limit=10, model_version="2")
    assert err.value.status_code == 503


def test_test_passenger_is_a_valid_request():
    main.Passenger(**main.TEST_PASSENGER.iloc[0].to_dict())

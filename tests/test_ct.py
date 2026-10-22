"""Checks for step 8 (continuous training) on fake data: the 7 standard candidates train, save and predict
like the API does, and the command line turns into the right training jobs."""

from pathlib import Path

import pytest

pytest.importorskip("torch")

from skysat import schema
from skysat.data import clean
from tests.test_data_and_drift import make_raw
from training.ct import CANDIDATES, parse_jobs
from training.train import load_bundle, model_file, predict_proba, run_training


@pytest.fixture(scope="module")
def data():
    df = clean(make_raw(n=900, seed=5))
    df[schema.TARGET] = ((df["online_boarding"] + df["seat_comfort"]) >= 6).astype(int)
    return df.iloc[:700], df.iloc[700:]


def test_every_candidate_trains_and_loads_like_the_api_does(data, tmp_path):
    assert len(CANDIDATES) == 7
    train, val = data
    for name, params in CANDIDATES.items():
        path, metrics = run_training(train, val, {**params, "epochs": 5, "batch_size": 128}, tmp_path / name)
        assert metrics["roc_auc"] > 0.8, name
        bundle = load_bundle(model_file(Path(path).parent))
        assert bundle["meta"]["params"]["model"] == params["model"]
        p = predict_proba(bundle, val)
        assert p.shape == (len(val),) and ((p >= 0) & (p <= 1)).all(), name
        # one passenger at a time (what /predict does) gives the same answer as the batch
        assert predict_proba(bundle, val.iloc[[0]])[0] == pytest.approx(p[0], abs=1e-5), name


def test_command_line_becomes_training_jobs():
    assert parse_jobs(["--all"]) == CANDIDATES
    assert parse_jobs(["--model", "random_forest"]) == {
        "random_forest": {"model": "random_forest", "rating_one_hot": False}}
    assert parse_jobs(["--model", "mlp", "--rating-one-hot", "--epochs", "30", "--name", "mlp-30"]) == {
        "mlp-30": {"model": "mlp", "rating_one_hot": True, "epochs": 30}}


@pytest.mark.parametrize("argv", [[], ["--all", "--model", "mlp"], ["--model", "catboost"],
                                  ["--model", "mlp", "--drop-features", "gendr"]])
def test_bad_command_lines_are_rejected(argv):
    with pytest.raises(SystemExit):
        parse_jobs(argv)

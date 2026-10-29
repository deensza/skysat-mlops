"""Checks for step 8 (continuous training) on fake data: the 8 standard candidates train, save and predict
like the API does, and the command line turns into the right training jobs."""

from pathlib import Path

import mlflow
import pytest

pytest.importorskip("torch")

from skysat import schema
from skysat.data import clean
from tests.test_data_and_drift import make_raw
from tests.test_track import local_mlflow  # noqa: F401 (fixture)
from training import track
from training.ct import CANDIDATES, parse_jobs, train_and_register
from training.train import load_bundle, model_file, predict_proba, run_training


@pytest.fixture(scope="module")
def data():
    df = clean(make_raw(n=900, seed=5))
    df[schema.TARGET] = ((df["online_boarding"] + df["seat_comfort"]) >= 6).astype(int)
    return df.iloc[:700], df.iloc[700:]


def test_every_candidate_trains_and_loads_like_the_api_does(data, tmp_path):
    assert len(CANDIDATES) == 8
    train, val = data
    on_demand = {m: {"model": m} for m in ("extra_trees", "decision_tree", "knn")}  # --model only
    for name, params in {**CANDIDATES, **on_demand}.items():
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


def test_runs_log_the_real_settings_the_training_curve_and_plots(data, local_mlflow):
    client, experiment = local_mlflow
    mlflow.set_experiment(experiment_id=experiment)
    train, val = data
    for name, params in {"lgbm": {"model": "lightgbm"}, "net": {"model": "mlp", "epochs": 3, "batch_size": 128}}.items():
        train_and_register(client, train, val, name, params, champion_auc=None)
    runs = {r.info.run_name: r for r in client.search_runs([experiment])}

    lgbm = runs["ct-lgbm"].data.params
    assert lgbm["model.n_estimators"] == "500" and "epochs" not in lgbm  # its own settings, not the MLP's
    net = runs["ct-net"]
    assert "epochs" in net.data.params and not any(k.startswith("model.") for k in net.data.params)
    assert [m.step for m in client.get_metric_history(net.info.run_id, "train_loss")] == [1, 2, 3]
    for run in runs.values():
        plots = {a.path for a in client.list_artifacts(run.info.run_id, "plots")}
        assert plots == {"plots/roc_curve.png", "plots/confusion_matrix.png"}

    # every run is also an MLflow 3 logged model (the experiment's Models tab), and the registry version points to
    # it: the champion downloads through that source exactly as the API does
    logged = client.search_logged_models(experiment_ids=[experiment])
    assert sorted(m.name for m in logged) == ["ct-lgbm", "ct-net"]
    assert all(m.metrics and m.params.get("model") for m in logged)
    champion = client.get_model_version_by_alias("skysat-satisfaction", "champion")
    assert champion.source.startswith("models:/m-")
    assert track.champion_auc_on(client, val) == pytest.approx(max(r.data.metrics["roc_auc"] for r in runs.values()))

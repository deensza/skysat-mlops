"""Checks for step 3a (training). Skipped automatically when PyTorch is not installed (e.g. in CI for now)."""

import numpy as np
import pytest

pytest.importorskip("torch")

from skysat import schema
from skysat.data import clean
from tests.test_data_and_drift import make_raw
from training.train import load_bundle, predict_proba, run_training, to_matrix

PARAMS = {"epochs": 8, "batch_size": 128, "hidden": [32, 16]}


@pytest.fixture(scope="module")
def data():
    """Fake clean passengers with a label the model can actually learn (good boarding + seat)."""
    df = clean(make_raw(n=900, seed=3))
    df[schema.TARGET] = ((df["online_boarding"] + df["seat_comfort"]) >= 6).astype(int)
    return df.iloc[:700], df.iloc[700:]


@pytest.fixture(scope="module")
def trained(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    path, metrics = run_training(data[0], data[1], PARAMS, out)
    return path, metrics


def test_run_training_returns_file_and_metrics(trained):
    path, metrics = trained
    assert path.endswith("model.pt")
    for key in ["roc_auc", "precision", "recall", "f1", "accuracy", "majority_baseline_accuracy"]:
        assert 0.0 <= metrics[key] <= 1.0


def test_model_learns_the_signal(trained):
    # AUC 0.5 would mean it learned nothing; a broken pipeline would sit near 0.5
    assert trained[1]["roc_auc"] > 0.85


def test_saved_model_reproduces_validation_metrics(trained, data):
    from sklearn.metrics import roc_auc_score
    bundle = load_bundle(trained[0])
    p = predict_proba(bundle, data[1])
    assert roc_auc_score(data[1][schema.TARGET], p) == pytest.approx(trained[1]["roc_auc"], abs=1e-4)


def test_single_row_equals_same_row_in_batch(trained, data):
    bundle = load_bundle(trained[0])
    batch = predict_proba(bundle, data[1])
    for i in (0, 5, 40):
        assert predict_proba(bundle, data[1].iloc[[i]])[0] == pytest.approx(batch[i], abs=1e-5)


def test_unseen_category_and_extra_columns_do_not_crash(trained, data):
    bundle = load_bundle(trained[0])
    row = data[1].iloc[[0]].copy()
    row["travel_class"] = "Premium Mystery"  # not in schema.CATEGORICAL_VALUES
    assert predict_proba(bundle, row).shape == (1,)  # also carries passenger_id and the label as extras


def test_one_hot_layout_does_not_depend_on_the_data(data):
    # a frame with only one class value still produces the full, fixed column layout
    only_eco = data[0].assign(travel_class="Eco")
    assert to_matrix(only_eco).shape == to_matrix(data[0]).shape


def test_scaling_uses_training_rows_only(trained, data):
    meta = load_bundle(trained[0])["meta"]
    n_cat = to_matrix(data[0]).shape[1] - len(meta["mean"])
    assert np.allclose(meta["mean"], to_matrix(data[0])[:, n_cat:].mean(0), atol=1e-4)

"""Checks for the model comparison. Skipped when PyTorch is not installed."""

import pytest

pytest.importorskip("torch")

from skysat import schema
from skysat.data import clean
from tests.test_data_and_drift import make_raw
from training.compare import compare, summarize


@pytest.fixture(scope="module")
def results():
    df = clean(make_raw(n=900, seed=5))
    df[schema.TARGET] = ((df["online_boarding"] + df["seat_comfort"]) >= 6).astype(int)  # learnable label
    # small data needs more epochs: the MLP is still under-trained after 6 (AUC 0.78), fine after 30 (0.998)
    return compare(df.iloc[:700], df.iloc[700:], seeds=[0, 1], params={"epochs": 30, "hidden": [32, 16], "batch_size": 64})


def test_every_model_is_scored_for_every_seed(results):
    assert len(results) == 4 * 2
    assert set(results["model"]) == {"majority class", "logistic regression", "gradient boosting", "PyTorch MLP"}


def test_majority_class_has_no_skill_and_real_models_do(results):
    means = summarize(results)["mean"]
    assert means["majority class"] == pytest.approx(0.5)  # constant prediction = AUC 0.5
    for name in ["logistic regression", "gradient boosting", "PyTorch MLP"]:
        assert means[name] > 0.85

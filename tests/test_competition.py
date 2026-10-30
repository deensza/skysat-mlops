"""Checks for the competition track (training/competition.py) on fake data shaped like the Kaggle files."""

import pandas as pd
import pytest

from tests.test_data_and_drift import make_raw
from training import competition


def competition_like(n: int, seed: int) -> pd.DataFrame:
    """Fake rows in the competition's format: no "Inflight service", a True/False target."""
    df = make_raw(n=n, seed=seed).drop(columns=["Inflight service"])
    df[competition.TARGET] = (df["Online boarding"] + df["Seat comfort"]) >= 6
    return df


@pytest.mark.parametrize("models", [["hgb"], ["lightgbm"], competition.MODELS])
def test_cross_validation_scores_every_fold_and_predicts_the_test_set(models):
    train = competition_like(600, seed=1)
    train[competition.TARGET] = train[competition.TARGET].astype(int)
    test = competition_like(200, seed=2).drop(columns=[competition.TARGET])
    scores, pred = competition.cross_validate(train, test, None, {"max_iter": 30}, folds=3, models=models)
    assert len(scores) == 3 and all(s > 0.9 for s in scores)  # the fake label is learnable
    assert len(pred) == len(test) and ((pred >= 0) & (pred <= 1)).all()


def test_original_dataset_is_mapped_to_the_competition_columns(tmp_path, monkeypatch):
    raw = make_raw(n=50, seed=3)  # the original's format: "Inflight service", text target
    raw.to_csv(tmp_path / "train.csv")  # with an index column, like the real file
    raw.to_csv(tmp_path / "test.csv")
    monkeypatch.setattr(competition, "DATA_DIR", tmp_path)
    columns = list(competition_like(5, seed=4).columns)

    original = competition.load_original(columns)
    assert list(original.columns) == columns and len(original) == 100
    assert set(original[competition.TARGET].unique()) <= {0, 1}

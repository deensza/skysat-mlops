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


@pytest.mark.parametrize("models", [["hgb"], ["lightgbm"], ["xgboost"], competition.DEFAULT_MODELS])
def test_cross_validation_scores_every_fold_and_predicts_the_test_set(models):
    train = competition_like(600, seed=1)
    train[competition.TARGET] = train[competition.TARGET].astype(int)
    test = competition_like(200, seed=2).drop(columns=[competition.TARGET])
    scores, pred, details = competition.cross_validate(train, test, None, {"max_iter": 30}, folds=3, models=models)
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


def test_original_rates_come_from_the_original_labels_only():
    original = competition_like(300, seed=4)
    original[competition.TARGET] = original[competition.TARGET].astype(int)
    rates = competition.original_rates(competition_like(50, seed=5), original)
    expected = original.groupby("Class")[competition.TARGET].mean()
    assert rates["Class rate"].tolist() == competition_like(50, seed=5)["Class"].map(expected).tolist()
    assert "Age rate by travel and class" in rates and "Class rate by travel and class" not in rates



def test_original_lookups_count_exact_matches_in_the_original():
    original = competition_like(300, seed=4)
    original[competition.TARGET] = original[competition.TARGET].astype(int)
    df = pd.concat([original.head(3), competition_like(2, seed=9).assign(**{"Flight Distance": -1})], ignore_index=True)
    lookups = competition.original_lookups(df, original)
    counts = original["Flight Distance"].value_counts()
    assert lookups["original distance count"].tolist()[:3] == df["Flight Distance"].head(3).map(counts).tolist()
    assert lookups["original distance count"].tolist()[3:] == [0, 0]  # unseen distance: no match, rate missing
    assert lookups["original distance rate"].iloc[3:].isna().all()


def test_target_encoding_never_sees_the_labels_it_is_scored_on():
    train = competition_like(400, seed=1)
    train[competition.TARGET] = train[competition.TARGET].astype(int)
    keys = competition.combo_keys(train)
    fit, held_out = keys.iloc[:300], keys.iloc[300:]
    y = train[competition.TARGET].iloc[:300]
    _, enc_held = competition.encode_combos(fit, y, [held_out])
    flipped = train[competition.TARGET].copy()
    flipped.iloc[300:] = 1 - flipped.iloc[300:]  # changing held-out labels must not change their encoding
    _, enc_again = competition.encode_combos(fit, flipped.iloc[:300], [held_out])
    pd.testing.assert_frame_equal(enc_held, enc_again)
    scores, _, _ = competition.cross_validate(train, competition_like(100, seed=2).drop(columns=[competition.TARGET]),
                                              None, {"max_iter": 30}, folds=3, target_encode=True)
    assert len(scores) == 3

def test_a_logged_run_has_scores_per_model_plots_tables_and_a_report(tmp_path, monkeypatch):
    import mlflow
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("skysat-kaggle", artifact_location=(tmp_path / "artifacts").as_uri())
    train = competition_like(600, seed=1)
    train[competition.TARGET] = train[competition.TARGET].astype(int)
    test = competition_like(200, seed=2).drop(columns=[competition.TARGET])
    scores, _, details = competition.cross_validate(train, test, None, {"max_iter": 30}, folds=3,
                                                    models=competition.MODELS, rates_from=train)
    competition.log_run({"model": "+".join(competition.MODELS)}, scores, details, train[competition.TARGET].to_numpy(), "abc")

    run = mlflow.search_runs(experiment_names=["skysat-kaggle"], output_format="list")[0]
    assert {"cv_roc_auc_mean", "oof_roc_auc", "hgb.cv_roc_auc_mean", "lightgbm.cv_roc_auc_mean",
            "xgboost.cv_roc_auc_mean"} <= set(run.data.metrics)
    files = {f.path for d in ("", "plots", "tables") for f in mlflow.MlflowClient().list_artifacts(run.info.run_id, d)}
    assert {"report.md", "plots/overview.png", "plots/folds.png", "plots/feature_importance.png",
            "tables/folds.csv", "tables/feature_importance.csv"} <= files
    assert not any("submission" in f for f in files)  # predictions on the competition's test rows stay local

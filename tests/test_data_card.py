"""Checks for the data card (training/data_card.py) on fake files: aggregates and hashes, never rows."""

import hashlib
import zipfile

from tests.test_data_and_drift import make_raw
from training import data_card


def fake_files(tmp_path):
    raw = make_raw(n=80, seed=4)
    raw.to_csv(tmp_path / "train.csv")  # with the original's unnamed index column
    with zipfile.ZipFile(tmp_path / "comp.zip", "w") as z:
        z.writestr("sample_submission.csv", "id,satisfaction\n1,0.44\n2,0.44\n3,0.44\n")
    return [("original", "train.csv", tmp_path / "train.csv", "http://example", "x", "y"),
            ("competition", "sample_submission.csv", tmp_path / "comp.zip", "http://example", "x", "y"),
            ("processed", "missing.csv", tmp_path / "missing.csv", "make data", "x", "y")]


def test_card_counts_hashes_and_skips_missing_files(tmp_path):
    cards, frames = data_card.build(fake_files(tmp_path))
    original, sample = cards
    assert len(cards) == 2 and set(frames) == {"original/train.csv", "competition/sample_submission.csv"}
    assert original["sha256"] == hashlib.sha256((tmp_path / "train.csv").read_bytes()).hexdigest()
    assert original["rows"] == 80 and "Unnamed: 0" not in original["schema"]
    assert original["schema"]["Class"]["values"]  # few distinct values: listed with counts
    assert "values" not in original["schema"]["Age"] and "min" in original["schema"]["Age"]
    assert sample["target"] is None and sample["satisfied_share"] is None  # a placeholder probability, not a label


def test_logged_run_has_the_card_and_no_rows(tmp_path, monkeypatch):
    import mlflow
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment(data_card.EXPERIMENT, artifact_location=(tmp_path / "artifacts").as_uri())
    data_card.log_run(*data_card.build(fake_files(tmp_path)))

    run = mlflow.search_runs(experiment_names=[data_card.EXPERIMENT], output_format="list")[0]
    assert run.data.metrics["original.train.rows"] == 80
    assert len(run.data.tags["original.train.sha256"]) == 64
    assert [f.path for f in mlflow.MlflowClient().list_artifacts(run.info.run_id)] == ["data_card.json", "data_card.md"]
    assert {d.dataset.name for d in run.inputs.dataset_inputs} == {"original/train.csv", "competition/sample_submission.csv"}

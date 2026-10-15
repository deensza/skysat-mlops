"""Stand-in for training.train.run_training, used by `python -m training.track --stub`.

Same interface: run_training(train_df, val_df, params) -> (model_path, metrics).
Returns fake metrics and a dummy model.pt, so the MLflow side can be tested without
the dataset or PyTorch (e.g. to try the registry and the champion alias).
"""

import tempfile
from pathlib import Path

import pandas as pd

FAKE_ROC_AUC = 0.96


def run_training(train_df: pd.DataFrame, val_df: pd.DataFrame, params: dict) -> tuple[Path, dict]:
    roc_auc = params.get("stub_roc_auc", FAKE_ROC_AUC)
    model_path = Path(tempfile.mkdtemp()) / "model.pt"
    model_path.write_bytes(b"dummy model for testing the MLflow logging\n")
    metrics = {
        "roc_auc": roc_auc,
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.93,
        "f1": 0.935,
    }
    return model_path, metrics

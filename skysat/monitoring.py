"""Drift measurement: Population Stability Index (PSI) per feature.

PSI compares how a feature is distributed in the training data (reference) and in recent
traffic (current):  PSI = sum over bins of (current% - reference%) * ln(current% / reference%).
Common reading: < 0.1 no change, 0.1-0.2 small change, >= 0.2 significant drift.
"""

import numpy as np
import pandas as pd

from skysat import schema

PSI_THRESHOLD = 0.2
EPS = 1e-4  # stands in for an empty bin, so the log is defined


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI of one feature. Text columns and the 0-5 ratings use their own values as bins;
    other numbers use the reference deciles (repeated edges, e.g. many zero delays, are merged)."""
    if reference.dtype == object or reference.nunique() <= 12:
        values = sorted(set(reference.unique()) | set(current.unique()), key=str)
        ref = reference.value_counts(normalize=True).reindex(values, fill_value=0).to_numpy()
        cur = current.value_counts(normalize=True).reindex(values, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref = np.histogram(reference, edges)[0] / len(reference)
        cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, EPS, None), np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def feature_psi(reference: pd.DataFrame, current: pd.DataFrame) -> pd.Series:
    """PSI of every model feature, highest first."""
    scores = {col: psi(reference[col], current[col]) for col in schema.FEATURES}
    return pd.Series(scores).sort_values(ascending=False)

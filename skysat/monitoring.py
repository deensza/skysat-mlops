"""Drift measurement: Population Stability Index (PSI) per feature.

PSI compares how a feature is distributed in the training data (reference) and in recent
traffic (current):  PSI = sum over bins of (current% - reference%) * ln(current% / reference%).
Common reading: < 0.1 no change, 0.1-0.2 small change, >= 0.2 significant drift.

A reference can be kept as a *profile*: per feature, its bins and the share of rows in each. A profile holds
no rows, only shares, so it can be logged to the public MLflow next to every model (drift_reference.json):
the drift check then compares traffic with what the serving model was last checked on, not always with the
original training data (otherwise traffic the model has already adapted to would count as drift forever).
"""

import numpy as np
import pandas as pd

from skysat import schema

PSI_THRESHOLD = 0.2
EPS = 1e-4  # stands in for an empty bin, so the log is defined
REFERENCE_FILE = "drift_reference.json"


def _plain(value):
    """numpy scalars -> plain Python, so a profile can be written as JSON."""
    return value.item() if hasattr(value, "item") else value


def column_profile(reference: pd.Series, bins: int = 10) -> dict:
    """Text columns and the 0-5 ratings use their own values as bins; other numbers use the reference
    deciles (repeated edges, e.g. many zero delays, are merged)."""
    if reference.dtype == object or reference.nunique() <= 12:
        shares = reference.value_counts(normalize=True)
        return {"values": [_plain(v) for v in shares.index], "shares": [float(s) for s in shares]}
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))[1:-1]  # inner edges only
    counts = np.histogram(reference, np.r_[-np.inf, edges, np.inf])[0]
    return {"edges": [float(e) for e in edges], "shares": [float(s) for s in counts / len(reference)]}


def psi_against(profile: dict, current: pd.Series) -> float:
    """PSI of one feature: current traffic against a column profile."""
    if "values" in profile:
        values = list(profile["values"]) + [v for v in current.unique() if v not in profile["values"]]
        ref = np.r_[profile["shares"], np.zeros(len(values) - len(profile["values"]))]
        cur = current.value_counts(normalize=True).reindex(values, fill_value=0).to_numpy()
    else:
        ref = np.asarray(profile["shares"])
        cur = np.histogram(current, np.r_[-np.inf, profile["edges"], np.inf])[0] / len(current)
    ref, cur = np.clip(ref, EPS, None), np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI of one feature, reference given as rows."""
    return psi_against(column_profile(reference, bins), current)


def profile(df: pd.DataFrame) -> dict:
    """Profile of every model feature: bins and shares, no rows."""
    return {"rows": len(df), "features": {col: column_profile(df[col]) for col in schema.FEATURES}}


def profile_psi(reference: dict, current: pd.DataFrame) -> pd.Series:
    """PSI of every model feature against a profile, highest first."""
    scores = {col: psi_against(reference["features"][col], current[col]) for col in schema.FEATURES}
    return pd.Series(scores).sort_values(ascending=False)


def feature_psi(reference: pd.DataFrame, current: pd.DataFrame) -> pd.Series:
    """PSI of every model feature, highest first."""
    return profile_psi(profile(reference), current)

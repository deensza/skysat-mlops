"""Fast checks for step 2 that run without the Kaggle files (used by CI)."""

import numpy as np
import pandas as pd
import pytest

from simulator.drift import AFFECTED_RATINGS, apply_storm
from simulator.simulate import to_request
from skysat import schema
from skysat.data import DataError, clean

CLEAN_TO_RAW = {v: k for k, v in schema.RAW_TO_CLEAN.items()}


def make_raw(n=200, seed=1):
    """A small fake dataset with the same columns and value ranges as the Kaggle files."""
    rng = np.random.default_rng(seed)
    data = {"id": np.arange(n)}
    for col, values in schema.CATEGORICAL_VALUES.items():
        data[CLEAN_TO_RAW[col]] = rng.choice(values, size=n)
    data["Age"] = rng.integers(7, 86, size=n)
    data["Flight Distance"] = rng.integers(31, 5000, size=n)
    for col in schema.RATING_FEATURES:
        data[CLEAN_TO_RAW[col]] = rng.integers(0, 6, size=n)
    data["Departure Delay in Minutes"] = rng.integers(0, 120, size=n)
    arrival = rng.integers(0, 120, size=n).astype(float)
    arrival[:5] = np.nan
    data["Arrival Delay in Minutes"] = arrival
    data["satisfaction"] = rng.choice(["satisfied", "neutral or dissatisfied"], size=n)
    return pd.DataFrame(data)


def test_clean_produces_project_schema():
    raw = make_raw()
    df = clean(raw)
    assert list(df.columns) == [schema.ID_COLUMN] + schema.FEATURES + [schema.TARGET]
    assert len(schema.FEATURES) == 22
    assert not df.isna().any().any()
    assert set(df[schema.TARGET].unique()) <= {0, 1}
    # missing arrival delays are filled with the departure delay
    assert (df["arrival_delay_minutes"][:5] == df["departure_delay_minutes"][:5]).all()


def test_clean_rejects_unknown_category():
    raw = make_raw()
    raw.loc[0, "Class"] = "First"
    with pytest.raises(DataError):
        clean(raw)


def test_storm_changes_features_but_not_labels():
    df = clean(make_raw(n=1000))
    out = apply_storm(df, share=0.6, seed=0)
    assert (out[schema.TARGET] == df[schema.TARGET]).all()
    assert out["departure_delay_minutes"].mean() > df["departure_delay_minutes"].mean() + 30
    for col in AFFECTED_RATINGS:
        assert out[col].mean() < df[col].mean()
        # 0 means "not rated" and must stay 0
        assert (out.loc[df[col] == 0, col] == 0).all()
    assert out[schema.RATING_FEATURES].min().min() >= schema.RATING_MIN
    assert out[schema.RATING_FEATURES].max().max() <= schema.RATING_MAX


def test_storm_is_reproducible():
    df = clean(make_raw())
    pd.testing.assert_frame_equal(apply_storm(df, seed=3), apply_storm(df, seed=3))


def test_request_has_features_but_no_label():
    row = clean(make_raw(n=1)).iloc[0]
    body = to_request(row)
    assert set(body) == {schema.ID_COLUMN, *schema.FEATURES}
    assert schema.TARGET not in body

"""PSI drift measurement (step 7), on fake data."""

import pandas as pd

from simulator.drift import apply_storm
from skysat.data import clean
from skysat.monitoring import PSI_THRESHOLD, feature_psi, psi
from tests.test_data_and_drift import make_raw


def test_same_distribution_has_no_drift():
    reference = clean(make_raw(n=4000, seed=1))
    current = clean(make_raw(n=2000, seed=2))
    assert feature_psi(reference, current).max() < 0.1


def test_storm_is_detected_on_delays_and_ratings():
    reference = clean(make_raw(n=4000, seed=1))
    current = apply_storm(clean(make_raw(n=2000, seed=2)), share=0.6, seed=0)
    scores = feature_psi(reference, current)
    assert scores["departure_delay_minutes"] >= PSI_THRESHOLD
    assert scores["inflight_wifi_service"] >= PSI_THRESHOLD
    assert scores["seat_comfort"] < 0.1  # not touched by the storm


def test_psi_handles_text_and_unseen_values():
    reference = pd.Series(["Eco"] * 50 + ["Business"] * 50, dtype=object)
    assert psi(reference, reference) == 0.0
    assert psi(reference, pd.Series(["Eco Plus"] * 100, dtype=object)) > 1.0

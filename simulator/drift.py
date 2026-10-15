"""Drift scenario: a winter storm.

A share of passengers is hit by the storm:
  - their departure is delayed by an extra 30-180 minutes,
  - the arrival delay follows the departure delay,
  - ratings that depend on the airport, the schedule and the overloaded online
    systems drop by 2 stars, but never below 1
    (a 0 rating means "not rated" and stays 0).

The labels (satisfied or not) are NOT changed: these passengers were satisfied or
not for their own reasons, but now they look different. A model trained before the
storm has never seen passengers like this, so its accuracy drops. Retraining on the
new data (step 7) fixes it.
"""

import numpy as np
import pandas as pd

from skysat import schema

AFFECTED_RATINGS = [
    "inflight_wifi_service",
    "departure_arrival_time_convenient",
    "gate_location",
    "checkin_service",
    "baggage_handling",
    "online_boarding",
    "ease_of_online_booking",
]
RATING_DROP = 2
EXTRA_DELAY_MINUTES = (30, 180)


def apply_storm(df: pd.DataFrame, share: float = 0.6, seed: int = 0) -> pd.DataFrame:
    """Return a copy of df where `share` of the passengers are hit by the storm."""
    if not 0.0 <= share <= 1.0:
        raise ValueError("share must be between 0 and 1")

    rng = np.random.default_rng(seed)
    out = df.copy()
    hit = rng.random(len(out)) < share
    n_hit = int(hit.sum())

    extra = rng.integers(EXTRA_DELAY_MINUTES[0], EXTRA_DELAY_MINUTES[1] + 1, size=n_hit)
    out.loc[hit, "departure_delay_minutes"] += extra
    noise = rng.integers(-10, 11, size=n_hit)
    out.loc[hit, "arrival_delay_minutes"] = np.maximum(
        out.loc[hit, "departure_delay_minutes"].to_numpy() + noise, 0
    )

    for col in AFFECTED_RATINGS:
        rated = hit & (out[col] > 0)
        out.loc[rated, col] = np.maximum(out.loc[rated, col] - RATING_DROP, 1)

    # keep the schema guarantees
    for col in schema.RATING_FEATURES:
        out[col] = out[col].clip(schema.RATING_MIN, schema.RATING_MAX).astype(int)
    out["departure_delay_minutes"] = out["departure_delay_minutes"].astype(int)
    out["arrival_delay_minutes"] = out["arrival_delay_minutes"].astype(int)
    return out

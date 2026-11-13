"""Replay live airline reviews into the complaint-box API, in date order, with the true labels.

    python -m simulator.replay --url http://localhost:8080/score --from 2019-01 --months 3 --per-month 50

For each month from --from, --per-month live reviews (flown >= 2019, all airlines) are posted one by one to
POST /score together with true_score and true_recommended, so the server can log how far off the model is.
The reviews are read from data/reviews/AirlineReviews.csv on this machine and only sent to the URL you give.
"""

import argparse
import base64
import json
import os
import urllib.error
import urllib.request

import pandas as pd

from llm.reviews import REVIEWS_FILE, load_reviews, split_reviews
from simulator.scenarios import SCENARIOS, apply_scenario


def month_start(text: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(pd.to_datetime(text, format="%Y-%m"))
    except ValueError as err:
        raise argparse.ArgumentTypeError("must be YYYY-MM, e.g. 2019-01") from err


def pick_reviews(live: pd.DataFrame, start: pd.Timestamp, months: int, per_month: int, seed: int = 0) -> pd.DataFrame:
    """per_month random live reviews for each of `months` months from `start`, in date order."""
    parts = []
    for k in range(months):
        first = start + pd.DateOffset(months=k)
        month = live[(live["date"] >= first) & (live["date"] < first + pd.DateOffset(months=1))]
        parts.append(month.sample(n=min(per_month, len(month)), random_state=seed + k))
    return pd.concat(parts).sort_values("date", kind="stable") if parts else live.iloc[:0]


def request_body(row) -> dict:
    return {"airline": row["AirlineName"], "cabin": row["CabinType"] if isinstance(row["CabinType"], str) else "",
            "travel_type": row["TravelType"] if isinstance(row["TravelType"], str) else "",
            "title": row["Title"] if isinstance(row["Title"], str) else "", "review": row["Review"],
            "true_score": int(row["score"]), "true_recommended": row["recommended"]}


def post(url: str, body: dict, timeout: float) -> bool:
    headers = {"Content-Type": "application/json"}
    user, password = os.environ.get("API_USER"), os.environ.get("API_PASSWORD")
    if user and password:
        headers["Authorization"] = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--url", required=True, help="the /score endpoint, e.g. http://localhost:8080/score")
    p.add_argument("--from", dest="start", type=month_start, default="2019-01", help="first month, YYYY-MM")
    p.add_argument("--months", type=int, default=3)
    p.add_argument("--per-month", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenario", default="none", choices=SCENARIOS, help="simulated drift (simulator/scenarios.py)")
    p.add_argument("--strength", type=float, default=None, help="scenario strength; default depends on the scenario")
    p.add_argument("--data", default=str(REVIEWS_FILE), help="the airline reviews CSV")
    p.add_argument("--timeout", type=float, default=120, help="seconds per request (CPU generation takes seconds)")
    args = p.parse_args()
    if isinstance(args.start, str):
        args.start = month_start(args.start)

    live = split_reviews(load_reviews(args.data), seed=args.seed)["live"]
    batch = pick_reviews(live, args.start, args.months, args.per_month, args.seed)
    batch = apply_scenario(batch, args.scenario, args.strength, seed=args.seed)  # text and true labels
    print(f"Replaying {len(batch)} live reviews, {batch['date'].min():%Y-%m} to {batch['date'].max():%Y-%m}"
          if len(batch) else "No live reviews in that range")
    ok = failed = 0
    for _, row in batch.iterrows():
        if post(args.url, request_body(row), args.timeout):
            ok += 1
        else:
            failed += 1
    print(f"ok: {ok}  failed: {failed}")


if __name__ == "__main__":
    main()

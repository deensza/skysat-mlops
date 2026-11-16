"""The simulated passengers: post reviews from the local dataset to POST /reviews (llm/intake.py), as passengers would,
each with the reviewer's own score, recommendation and sub-ratings.

    python -m simulator.replay --url http://localhost:8001 --seed-train 2000          # the first model's training data
    python -m simulator.replay --url http://localhost:8001 --from 2019-01 --months 1 --per-month 100   # live passengers
    python -m simulator.replay --url http://localhost:8001 --from 2019-02 --months 1 --per-month 100 --scenario harsh

--seed-train posts reviews flown <= 2018 (not the held-out airlines) with "live": false: they go into the queue for
the trainer but are not sent to the model as traffic. The live mode posts reviews flown >= 2019 in date order; the
intake queues them and sends each one to the model with its true labels (the dashboard, the drift check), and the
next training batch is made of them. --scenario changes the text and labels first (simulator/scenarios.py): `harsh`
is a new survey that scores everything lower, the drift the current model misses.

The reviews are read from data/reviews/AirlineReviews.csv on this machine and only sent to the URL you give. Each
review has a fixed id (kaggle128k:<row>[:scenario]), so posting the same one twice adds it once.
"""

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request

import pandas as pd

from llm.reviews import REVIEWS_FILE, load_reviews, rated_values, split_reviews
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


def _text(value) -> str:
    return value if isinstance(value, str) else ""


def request_body(row, scenario: str = "none", live: bool = True) -> dict:
    """One review as a passenger would send it to POST /reviews."""
    key = row.get("unique_id", row.name)
    return {"review_id": f"kaggle128k:{key}" + ("" if scenario == "none" else f":{scenario}"),
            "airline": _text(row["AirlineName"]) or "unknown", "cabin": _text(row.get("CabinType")),
            "travel_type": _text(row.get("TravelType")), "title": _text(row.get("Title")), "review": row["Review"],
            "flown": row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else None,
            "score": int(row["score"]), "recommended": row["recommended"], "ratings": rated_values(row),
            "source": "simulator", "live": live}


def post(url: str, body: dict, timeout: float = 30) -> bool:
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
    p.add_argument("--url", required=True, help="the intake (or the gateway in front of it), e.g. http://localhost:8001")
    p.add_argument("--seed-train", type=int, default=0, metavar="N",
                   help="post N training reviews (flown <= 2018) for the first model instead of live ones")
    p.add_argument("--from", dest="start", type=month_start, default="2019-01", help="first live month, YYYY-MM")
    p.add_argument("--months", type=int, default=1)
    p.add_argument("--per-month", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenario", default="none", choices=SCENARIOS, help="simulated drift (simulator/scenarios.py)")
    p.add_argument("--strength", type=float, default=None, help="scenario strength; default depends on the scenario")
    p.add_argument("--rate", type=float, default=10, help="most reviews posted per second (the gateway allows 20)")
    p.add_argument("--data", default=str(REVIEWS_FILE), help="the airline reviews CSV")
    args = p.parse_args()
    if isinstance(args.start, str):
        args.start = month_start(args.start)

    parts = split_reviews(load_reviews(args.data), seed=args.seed)
    if args.seed_train:
        batch, live = parts["train"].sample(n=min(args.seed_train, len(parts["train"])), random_state=args.seed), False
    else:
        batch, live = pick_reviews(parts["live"], args.start, args.months, args.per_month, args.seed), True
    batch = apply_scenario(batch, args.scenario, args.strength, seed=args.seed)  # text and true labels
    if batch.empty:
        print("No reviews in that range")
        return
    kind = "training" if not live else "live"
    print(f"Posting {len(batch)} {kind} reviews flown {batch['date'].min():%Y-%m} to {batch['date'].max():%Y-%m}"
          f" (scenario {args.scenario}) to {args.url}/reviews")
    ok = failed = 0
    url = f"{args.url.rstrip('/')}/reviews"
    for _, row in batch.iterrows():
        started = time.time()
        if post(url, request_body(row, args.scenario, live)):
            ok += 1
        else:
            failed += 1
        time.sleep(max(0.0, 1 / args.rate - (time.time() - started)))
    print(f"ok: {ok}  failed: {failed}")


if __name__ == "__main__":
    main()

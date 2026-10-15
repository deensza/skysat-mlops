"""Step 2 – Traffic simulator.

Takes passengers from data/processed/live.csv (never used for training) and either
writes them to a CSV file or sends them to the prediction API one by one.

Modes:
    normal  passengers as they are
    drift   passengers after a winter storm (see simulator/drift.py)

Examples (run from the project root):
    python -m simulator.simulate generate --mode normal --n 2000 --out data/sim_normal.csv
    python -m simulator.simulate generate --mode drift  --n 2000 --out data/sim_drift.csv
    python -m simulator.simulate compare --n 5000
    python -m simulator.simulate generate --mode drift --n 500 --url http://localhost:8080/predict
"""

import argparse
import json
import time
import urllib.error
import urllib.request

import pandas as pd

from simulator.drift import AFFECTED_RATINGS, apply_storm
from skysat import schema
from skysat.data import PROCESSED_DIR

LIVE_FILE = PROCESSED_DIR / "live.csv"


def load_live() -> pd.DataFrame:
    if not LIVE_FILE.exists():
        raise SystemExit(f"ERROR: {LIVE_FILE} not found. Run `python -m training.prepare_data` first.")
    return pd.read_csv(LIVE_FILE)


def make_batch(mode: str, n: int, seed: int, share: float) -> pd.DataFrame:
    live = load_live()
    batch = live.sample(n=min(n, len(live)), random_state=seed, replace=False).reset_index(drop=True)
    if mode == "drift":
        batch = apply_storm(batch, share=share, seed=seed)
    return batch


def to_request(row: pd.Series) -> dict:
    """The JSON body the API receives: the passenger id + the model features (no label)."""
    body = {schema.ID_COLUMN: int(row[schema.ID_COLUMN])}
    for col in schema.FEATURES:
        value = row[col]
        body[col] = value if col in schema.CATEGORICAL_FEATURES else int(value)
    return body


def send(batch: pd.DataFrame, url: str, rate: float) -> None:
    ok = failed = 0
    started = time.perf_counter()
    for _, row in batch.iterrows():
        data = json.dumps(to_request(row)).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                ok += 200 <= resp.status < 300
        except (urllib.error.URLError, TimeoutError) as err:
            failed += 1
            if failed == 1:
                print(f"  first error: {err}")
            if ok == 0 and failed >= 5:
                raise SystemExit(f"ERROR: the API at {url} is not reachable. Is it running?")
        if rate > 0:
            time.sleep(1.0 / rate)
    took = time.perf_counter() - started
    print(f"Sent {len(batch)} requests to {url} in {took:.1f}s: {ok} ok, {failed} failed")


def describe(batch: pd.DataFrame) -> dict:
    return {
        "departure delay (avg min)": batch["departure_delay_minutes"].mean(),
        "arrival delay (avg min)": batch["arrival_delay_minutes"].mean(),
        "delayed > 30 min (%)": (batch["departure_delay_minutes"] > 30).mean() * 100,
        **{f"{c} (avg)": batch[c].mean() for c in AFFECTED_RATINGS},
        "satisfied (%)": batch[schema.TARGET].mean() * 100,
    }


def cmd_generate(args):
    batch = make_batch(args.mode, args.n, args.seed, args.share)
    if args.url:
        send(batch, args.url, args.rate)
    else:
        out = args.out or f"data/sim_{args.mode}.csv"
        batch.to_csv(out, index=False)
        print(f"Wrote {len(batch)} {args.mode} passengers to {out}")


def cmd_compare(args):
    normal = describe(make_batch("normal", args.n, args.seed, args.share))
    drift = describe(make_batch("drift", args.n, args.seed, args.share))
    print(f"Same {args.n} passengers, before and after the storm (share hit: {args.share:.0%})\n")
    width = max(len(k) for k in normal) + 2
    print(f"  {'':{width}}{'normal':>10}{'drift':>10}")
    for key in normal:
        print(f"  {key:{width}}{normal[key]:>10.2f}{drift[key]:>10.2f}")


def share_value(text: str) -> float:
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1, e.g. 0.6")
    return value


def main():
    parser = argparse.ArgumentParser(description="SkySat traffic simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--n", type=int, default=1000, help="number of passengers (default 1000)")
    common.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    common.add_argument("--share", type=share_value, default=0.6, help="drift: share of passengers hit (default 0.6)")

    gen = sub.add_parser("generate", parents=[common], help="write passengers to a CSV or send them to the API")
    gen.add_argument("--mode", choices=["normal", "drift"], default="normal")
    gen.add_argument("--out", help="output CSV (default data/sim_<mode>.csv)")
    gen.add_argument("--url", help="send each passenger as a POST request to this URL instead of writing a file")
    gen.add_argument("--rate", type=float, default=0, help="requests per second when sending (0 = as fast as possible)")
    gen.set_defaults(func=cmd_generate)

    cmp_ = sub.add_parser("compare", parents=[common], help="show how the drift changes the data")
    cmp_.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

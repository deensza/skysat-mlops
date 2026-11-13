"""Drift check on the live traffic: reads GET /scores/recent from the server and compares the model's real accuracy
with the gate (llm/gate.py).

    python -m llm.drift_check --url http://localhost:8080 --hours 24

DRIFT if, on at least --min-labelled requests that came with the true answer, the live MAE is above gate.MAE_MAX or
the live recommended accuracy is below gate.RECOMMENDED_MIN. Prints a markdown table. Exit code is always 0 (a failed
read is "no drift, not enough data"); with $GITHUB_OUTPUT set it appends `drift=true` or `drift=false`.

The server answers with aggregates only (counts, means, shares), never review text. If it sits behind the gateway's
login, set API_USER and API_PASSWORD. The URL can also come from SKYSAT_SERVER (an address without scheme).
"""

import argparse
import base64
import json
import os
import urllib.error
import urllib.request

from llm import gate

MIN_LABELLED = 50


def fetch(base_url: str, hours: float, timeout: float = 30) -> dict:
    headers = {}
    user, password = os.environ.get("API_USER"), os.environ.get("API_PASSWORD")
    if user and password:
        headers["Authorization"] = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
    url = f"{base_url.rstrip('/')}/scores/recent?hours={hours:g}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as resp:
        return json.load(resp)


def decide(summary: dict, min_labelled: int = MIN_LABELLED) -> tuple[bool, list[tuple[str, str, str, str]]]:
    """(drift, rows of (measure, live value, limit, verdict)) from a /scores/recent answer."""
    rows, drift = [], False
    checks = [("score MAE", "mae", "labelled_score", f"<= {gate.MAE_MAX}", lambda v: v > gate.MAE_MAX),
              ("recommended accuracy", "recommended_accuracy", "labelled_recommended", f">= {gate.RECOMMENDED_MIN}",
               lambda v: v < gate.RECOMMENDED_MIN)]
    for name, key, labelled_key, limit, bad in checks:
        value, labelled = summary.get(key), summary.get(labelled_key, 0)
        if value is None or labelled < min_labelled:
            verdict = f"not enough labelled requests ({labelled} < {min_labelled})"
        elif bad(value):
            verdict, drift = "DRIFT", True
        else:
            verdict = "ok"
        if verdict in ("ok", "DRIFT"):
            verdict += f" (n={labelled})"
        rows.append((name, "-" if value is None else f"{value:.3f}", limit, verdict))
    return drift, rows


def markdown(summary: dict, drift: bool, rows) -> str:
    lines = [f"## Drift check: {'DRIFT' if drift else 'no drift'}", "",
             f"Last {summary.get('hours', 0):g} h: {summary.get('count', 0)} requests, "
             f"mean predicted score {summary.get('mean_score')}", "",
             "| Measure | Live | Gate limit | Verdict |", "|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    shares = summary.get("complaint_share") or {}
    if shares:
        lines += ["", "| Complaint | Share |", "|---|---|"] + [f"| {k} | {v:.1%} |" for k, v in shares.items()]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    default_url = f"https://{os.environ['SKYSAT_SERVER']}" if os.environ.get("SKYSAT_SERVER") else None
    p.add_argument("--url", default=os.environ.get("SKYSAT_URL") or default_url,
                   help="the server's base URL (or env SKYSAT_URL / SKYSAT_SERVER)")
    p.add_argument("--hours", type=float, default=24)
    p.add_argument("--min-labelled", type=int, default=MIN_LABELLED)
    args = p.parse_args()
    if not args.url:
        raise SystemExit("Give --url, or set SKYSAT_URL / SKYSAT_SERVER")

    try:
        summary = fetch(args.url, args.hours)
        drift, rows = decide(summary, args.min_labelled)
        report = markdown(summary, drift, rows)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as err:
        drift, report = False, f"## Drift check: no drift\n\nCould not read {args.url}/scores/recent: {err}"
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report + "\n")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"drift={'true' if drift else 'false'}\n")


if __name__ == "__main__":
    main()

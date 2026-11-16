"""Drift check on the live traffic: reads GET /scores/recent from the server and compares the model's real accuracy
with the gate (llm/gate.py).

    python -m llm.drift_check --url http://localhost:8080 --hours 24

DRIFT if, on at least --min-labelled requests that came with the true answer, the live MAE is above gate.MAE_MAX or
the live recommended accuracy is below gate.RECOMMENDED_MIN, by more than NOISE_SE standard errors of the live sample.
Prints a markdown table. Exit code 0 when the server was read (with $GITHUB_OUTPUT set it appends `drift=true` or
`drift=false`); exit code 2 when it could not be read or understood (the report starts with "Drift check: FAILED" and
no `drift=` line is written).

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

MIN_LABELLED = 100
# A measure counts as DRIFT only when it is worse than the gate limit by more than NOISE_SE standard errors of the
# live sample, so a healthy model near the limit is not flagged by sampling noise.
NOISE_SE = 1.0


def fetch(base_url: str, hours: float, timeout: float = 30) -> dict:
    headers = {}
    user, password = os.environ.get("API_USER"), os.environ.get("API_PASSWORD")
    if user and password:
        headers["Authorization"] = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
    url = f"{base_url.rstrip('/')}/scores/recent?hours={hours:g}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as resp:
        return json.load(resp)


def decide(summary: dict, min_labelled: int = MIN_LABELLED) -> tuple[bool, list[tuple[str, str, str, str]]]:
    """(drift, rows of (measure, live value ± standard error, limit, verdict)) from a /scores/recent answer."""
    rows, drift = [], False
    mae_n, acc_n = summary.get("labelled_score", 0), summary.get("labelled_recommended", 0)
    mae, acc = summary.get("mae"), summary.get("recommended_accuracy")
    mae_sd = summary.get("mae_sd")  # missing on an older server: plain comparison
    mae_se = mae_sd / mae_n ** 0.5 if mae_sd is not None and mae_n else 0.0
    acc_se = (acc * (1 - acc) / acc_n) ** 0.5 if acc is not None and acc_n else 0.0
    checks = [("score MAE", mae, mae_n, mae_se, f"<= {gate.MAE_MAX}", lambda v, se: v - NOISE_SE * se > gate.MAE_MAX),
              ("recommended accuracy", acc, acc_n, acc_se, f">= {gate.RECOMMENDED_MIN}",
               lambda v, se: v + NOISE_SE * se < gate.RECOMMENDED_MIN)]
    for name, value, labelled, se, limit, bad in checks:
        if value is None or labelled < min_labelled:
            verdict = f"not enough labelled requests ({labelled} < {min_labelled})"
        elif bad(value, se):
            verdict, drift = "DRIFT", True
        else:
            verdict = "ok"
        if verdict in ("ok", "DRIFT"):
            verdict += f" (n={labelled})"
        shown = "-" if value is None else f"{value:.3f}" + (f" ± {se:.3f}" if se else "")
        rows.append((name, shown, limit, verdict))
    return drift, rows


def markdown(summary: dict, drift: bool, rows) -> str:
    lines = [f"## Drift check: {'DRIFT' if drift else 'no drift'}", "",
             f"Last {summary.get('hours', 0):g} h: {summary.get('count', 0)} requests, "
             f"mean predicted score {summary.get('mean_score')}", "",
             "| Measure | Live (± 1 standard error) | Gate limit | Verdict |", "|---|---|---|---|"]
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
        drift, report = None, f"## Drift check: FAILED\n\nCould not read {args.url}/scores/recent: {err}"
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report + "\n")
    if drift is None:
        raise SystemExit(2)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"drift={'true' if drift else 'false'}\n")


if __name__ == "__main__":
    main()

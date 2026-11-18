"""Worker 2, the trainer: takes a batch of reviews from the queue, LoRA-tunes, gates, logs and registers.

    python -m llm.trainer --queue-url https://<server>        # login: QUEUE_USER / QUEUE_PASSWORD
    python -m llm.trainer --queue-url http://localhost:8001 --no-mlflow --n 300   # a local trial: batch goes back

Runs where there is a GPU: a RunPod pod started by llm/runpod_pod.py (which deletes the pod when this exits), or a
local NVIDIA card. Steps:
  1. POST /queue/claim: up to --n pending reviews that the model has already scored as live traffic (the most
     recently received first; training-only reviews count as scored) and --replay already-trained ones
  2. 10% of the NEW reviews, at least --n-gate-min, are kept back for the gate (never trained on); the rest plus the
     replayed ones train
  3. llm/train_lora.py run(): train, gate, compare with the champion on the same gate reviews, log, register, promote
  4. POST /queue/done ok=true (the batch is trained) only if a version was registered; ok=false otherwise (the gate
     failed, --no-mlflow, --keep-pending, any error): the batch goes back to pending for the next run
Fewer than --min-new pending reviews: nothing is trained and the batch goes straight back.
"""

import argparse
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from llm.reviews import GATE_SHARE, build_examples, from_queue, split_batch
from llm.train_lora import BASE_MODEL, DEFAULT_OUT, run


def call(base_url: str, path: str, body: dict | None = None, timeout: float = 120) -> dict:
    headers = {"Content-Type": "application/json"}
    user, password = os.environ.get("QUEUE_USER"), os.environ.get("QUEUE_PASSWORD")
    if user and password:
        headers["Authorization"] = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"{base_url.rstrip('/')}{path}", data=data, headers=headers,
                                 method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def prepare(batch: dict, n_gate_max: int, seed: int, n_gate_min: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training examples, gate examples). The gate is drawn only from the batch's new reviews, and no training row
    has the same text as a gate row. The gate is 10% of the new reviews, but at least n_gate_min of them (never more
    than half): on 20 reviews the gate's limits would be decided by chance."""
    key = lambda df: df["Review"].str.lower().str.split().str.join(" ")
    new = from_queue(batch["reviews"])
    new = new[~key(new).duplicated()].reset_index(drop=True)
    replay = from_queue(batch["replay"])
    train_new, gate_new = split_batch(new, seed=seed, share=min(0.5, max(GATE_SHARE, n_gate_min / len(new))))
    gate_rows = build_examples(gate_new, min(n_gate_max, len(gate_new)), seed=seed + 1000)
    train_all = pd.concat([train_new, replay], ignore_index=True)
    train_all = train_all[~key(train_all).isin(set(key(gate_rows)))]
    return build_examples(train_all, seed=seed), gate_rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--queue-url", default=os.environ.get("QUEUE_URL"), help="the intake, or the gateway in front of it")
    p.add_argument("--n", type=int, default=2000, help="most new reviews to take from the queue")
    p.add_argument("--replay", type=int, default=500, help="already-trained reviews mixed in")
    p.add_argument("--min-new", type=int, default=300, help="fewer new reviews than this: train nothing")
    p.add_argument("--n-gate-min", type=int, default=100, help="fewest gate reviews (10%% of the new ones if that is more)")
    p.add_argument("--n-gate-max", type=int, default=200, help="most gate reviews (each is generated twice)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=2)  # fits an 8 GB card with gradient checkpointing
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where the adapter is saved")
    p.add_argument("--max-train-seconds", type=float, default=None,
                   help="stop training after this many seconds and go on to the gate and the registration; default: no limit. "
                        "The training part is only 80-92 s of a 6.5-10 min pod run, so this saves at most about a minute; "
                        "a cut adapter is undertrained")
    p.add_argument("--no-mlflow", action="store_true", help="train and gate only: register nothing (the batch goes back)")
    p.add_argument("--no-promote", action="store_true", help="register, but leave the champion alias where it is")
    p.add_argument("--keep-pending", action="store_true",
                   help="put the batch back even if a version was registered (trials must not use up the queue)")
    args = p.parse_args(argv)
    if not args.queue_url:
        p.error("give --queue-url or set QUEUE_URL")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    print(f"Queue before: {call(args.queue_url, '/queue/stats')}")
    batch = call(args.queue_url, "/queue/claim", {"n": args.n, "replay": args.replay})
    batch_id = batch["batch_id"]
    print(f"Batch {batch_id}: {len(batch['reviews'])} new reviews, {len(batch['replay'])} replayed")
    ok = False
    try:
        if len(batch["reviews"]) < args.min_new:
            print(f"Fewer than --min-new {args.min_new} new reviews: nothing to train")
            return 0
        train_rows, gate_rows = prepare(batch, args.n_gate_max, args.seed, args.n_gate_min)
        print(f"{len(train_rows)} training reviews, {len(gate_rows)} gate reviews")
        result = run(train_rows, gate_rows, args, params={"batch_id": batch_id, "n_new": len(batch["reviews"]),
                                                          "n_replay": len(batch["replay"])})
        print(f"Result: registered={result['registered']} version={result['version']} promoted={result['promoted']}")
        ok = bool(result["registered"]) and not args.keep_pending
        if not result["registered"]:
            why = "--no-mlflow: nothing is registered" if args.no_mlflow else "the gate failed"
            print(f"No version registered ({why}): the batch goes back to pending")
        return 0
    finally:
        if batch["reviews"]:
            print(f"Queue: {call(args.queue_url, '/queue/done', {'batch_id': batch_id, 'ok': ok})}")


if __name__ == "__main__":
    sys.exit(main())

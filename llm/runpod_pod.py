"""Starts the trainer (llm/trainer.py) on the cheapest RunPod GPU that is free, and makes sure the pod goes away.

    RUNPOD_API_KEY=... python -m llm.runpod_pod --wait       # start, follow the log, delete the pod at the end
    python -m llm.runpod_pod --dry-run                       # print the request, start nothing
    python -m llm.runpod_pod --sweep                         # delete trainer pods left behind (exit 1 if any)

With --wait the exit code is the trainer's, read from the pod's log. If the log cannot be read (RunPod answers 403 to
an API key that may create pods but not read logs), the outcome is taken from the server instead: 0 if the batch was
marked trained (a version was registered), else 1. With $GITHUB_OUTPUT set it writes pod_id (as soon as a pod exists),
exit_code, registered, version and promoted for the workflow (.github/workflows/train.yml).

The pod clones this repo's --branch from GitHub (the code must be pushed), installs transformers/peft/mlflow on the
image's own PyTorch, downloads the base model, runs the trainer against QUEUE_URL and deletes itself when the trainer
exits. Two more guards against a forgotten pod: a timer inside the pod deletes it after --max-minutes, and with
--wait this script deletes it too once --max-minutes have passed.

GPUs are tried cheapest first (community cloud prices read from RunPod's catalog on 2026-11-16, $/hour):
RTX 3070 8 GB 0.13, RTX A4000 16 GB 0.17, RTX 3080 Ti 12 GB 0.18, RTX A4500 20 GB 0.19, RTX 4000 Ada 20 GB 0.20,
RTX 3090 24 GB 0.22, and as a last resort RTX 4090 24 GB 0.34 (price of 2026-11-17): on that day all six cheaper
types were out of stock at once and a training started by the drift check rented nothing. A 1.5B model with LoRA and
gradient checkpointing fits 8 GB (measured on an RTX 4070 8 GB).

Env passed to the pod (from this process's env): QUEUE_URL, QUEUE_USER, QUEUE_PASSWORD (the gateway's `trainer`
login), MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD (DagsHub). They are visible in the pod's settings on RunPod.
RunPod gives every pod RUNPOD_POD_ID and a key scoped to that pod, which the pod uses to delete itself.

With COMMENT_SCRIPT set (the Train workflow sets it to .github/scripts/train-comment.sh) the launcher posts a comment
on the GitHub issue "Training runs" when the pod is rented, when the MLflow run shows up and when the pod is gone. The
comments come from this process only: the pod never gets the GitHub token, and a comment that fails is logged and
never ends the run. Without COMMENT_SCRIPT (make train-remote) nothing is posted.
"""

import argparse
import base64
import http.client
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

API = "https://api.runpod.io/v2"
IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
GPUS = ["NVIDIA GeForce RTX 3070", "NVIDIA RTX A4000", "NVIDIA GeForce RTX 3080 Ti", "NVIDIA RTX A4500",
        "NVIDIA RTX 4000 Ada Generation", "NVIDIA GeForce RTX 3090",
        "NVIDIA GeForce RTX 4090"]  # the last one only when none of the cheaper ones is free
# the catalog prices of the docstring, for a pod object that does not carry its own `cost`
PRICES = {"NVIDIA GeForce RTX 3070": 0.13, "NVIDIA RTX A4000": 0.17, "NVIDIA GeForce RTX 3080 Ti": 0.18,
          "NVIDIA RTX A4500": 0.19, "NVIDIA RTX 4000 Ada Generation": 0.20, "NVIDIA GeForce RTX 3090": 0.22,
          "NVIDIA GeForce RTX 4090": 0.34}
REPO = "https://github.com/deensza/skysat-mlops.git"
PASS_ENV = ["QUEUE_URL", "QUEUE_USER", "QUEUE_PASSWORD",
            "MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]
PACKAGES = "transformers==5.20.2 peft==0.21.2 mlflow-skinny==3.17.1 pandas==2.2.3"
POD_NAME = "skysat-trainer"
DEAD = ("EXITED", "ERROR", "TERMINATED")  # pod states (RunPod's `status`) in which the container no longer runs
# what a request can raise when the network or the answer breaks (IncompleteRead is not an OSError)
NETWORK = (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException)

# Runs inside the pod. Every way out ends in delete_pod; the background timer is the last resort.
POD_SCRIPT = r"""
delete_pod() {
  echo "deleting pod $RUNPOD_POD_ID"
  runpodctl pod delete "$RUNPOD_POD_ID" >/dev/null 2>&1 || runpodctl remove pod "$RUNPOD_POD_ID" >/dev/null 2>&1 \
    || curl -sf -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" \
    || echo "POD DELETE FAILED: delete pod $RUNPOD_POD_ID by hand in the RunPod console"
}
( sleep $((MAX_MINUTES * 60)); echo "TIME LIMIT of $MAX_MINUTES minutes reached"; delete_pod ) &
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p /workspace && cd /workspace && git clone -q --depth 1 -b "$BRANCH" "$REPO" code && cd code && git log -1 --oneline \
  && pip install -q --root-user-action=ignore $PACKAGES \
  && GIT_COMMIT=$(git rev-parse HEAD) python -u -m llm.trainer $TRAINER_ARGS
echo "trainer exit code $?"
sleep 90  # the launcher reads this log about every 40 s: give it time to see the last lines before the pod is gone
delete_pod
sleep 600
"""


def missing_settings() -> list[str]:
    """The environment variables a launch needs and does not find."""
    names = ("RUNPOD_API_KEY", "QUEUE_URL", "QUEUE_USER", "QUEUE_PASSWORD")
    return [k for k in names if not os.environ.get(k)]


def request(method: str, path: str, body: dict | None = None, timeout: float = 60):
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("Set RUNPOD_API_KEY (RunPod console > Settings > API Keys)")
    req = urllib.request.Request(f"{API}{path}", method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else {}


def pod_body(gpu: str, branch: str, trainer_args: str, max_minutes: int, name: str = POD_NAME) -> dict:
    env = {k: os.environ[k] for k in PASS_ENV if os.environ.get(k)}
    env |= {"BRANCH": branch, "REPO": REPO, "TRAINER_ARGS": trainer_args, "MAX_MINUTES": str(max_minutes),
            "PACKAGES": PACKAGES, "HF_HOME": "/workspace/hf", "PYTHONUNBUFFERED": "1"}
    return {"name": name, "image": IMAGE, "cloud": "COMMUNITY", "disk": 30, "env": env,
            "gpu": {"id": gpu, "count": 1, "minCudaVersion": "12.8"},
            "entrypoint": ["/bin/bash", "-c"], "cmd": [POD_SCRIPT]}


def launch(gpus=None, branch: str | None = None, trainer_args: str | None = None, max_minutes: int = 90,
           log=print) -> dict:
    """Creates the pod on the first GPU type that has capacity. Returns the pod as RunPod describes it."""
    missing = [k for k in missing_settings() if k.startswith("QUEUE_")]  # request() checks the API key
    if missing:
        raise SystemExit(f"Set {', '.join(missing)}: the pod needs them to reach the queue")
    branch = branch or os.environ.get("TRAIN_BRANCH", "main")
    trainer_args = trainer_args if trainer_args is not None else os.environ.get("TRAINER_ARGS", "")
    errors = []
    for gpu in gpus or GPUS:
        try:
            pod = request("POST", "/pods", pod_body(gpu, branch, trainer_args, max_minutes))
        except urllib.error.HTTPError as err:
            errors.append(f"{gpu}: {err.code} {err.read().decode(errors='replace')[:200]}")
            if err.code in (401, 403):  # not a capacity problem: another GPU type will not help
                raise RuntimeError(f"RunPod refused the API key: {errors[-1]}") from None
            log(f"No {gpu}: {errors[-1]}")
            continue
        except (*NETWORK, ValueError) as err:
            # the answer was lost: the pod may exist without this process knowing its id
            log(f"Creating the pod failed without an answer ({err}): deleting any trainer pod that was made")
            sweep(log)
            raise RuntimeError(f"Could not reach RunPod to create the pod: {err}") from None
        log(f"Pod {pod.get('id')} created on {gpu} (branch {branch}, trainer args '{trainer_args}')")
        if pod.get("cost") is not None:
            price = f"${pod['cost']}/h"
        elif gpu in PRICES:
            price = f"${PRICES[gpu]:.2f}/h (list price of 2026-11-16)"
        else:
            price = "price unknown"
        comment(f"A GPU was rented: pod {pod.get('id')}, {gpu}, {price}. The trainer clones main, downloads the base "
                f"model and trains on the newest ready reviews.", log)
        return pod
    raise RuntimeError("No GPU had capacity:\n" + "\n".join(errors))


def comment(text: str, log=print) -> None:
    """Posts text as a comment on the "Training runs" issue through $COMMENT_SCRIPT. Never raises: a comment that
    fails is one log line, and the training goes on. Put no secret and no env value in the text."""
    script = os.environ.get("COMMENT_SCRIPT")
    if not script:
        return
    try:
        done = subprocess.run(["bash", script, text], timeout=60, capture_output=True, text=True)
        if done.returncode:
            last = (done.stderr.strip().splitlines() or ["no error text"])[-1]
            log(f"Comment not posted: exit code {done.returncode}, {last}")
    except Exception as err:  # a missing script, a timeout, anything
        log(f"Comment not posted: {type(err).__name__}: {err}")


def mlflow_new_run(since_ms: int) -> str | None:
    """The MLflow UI link of the newest run started after since_ms (milliseconds) in the trainer's experiment, or
    None (no run yet, MLFLOW_TRACKING_URI is not a server, or it did not answer). Stdlib only: this runs on the
    workflow's bare python3."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").rstrip("/")
    if not uri.startswith("http"):
        return None
    login = base64.b64encode(f"{os.environ.get('MLFLOW_TRACKING_USERNAME')}:"
                             f"{os.environ.get('MLFLOW_TRACKING_PASSWORD')}".encode()).decode()
    headers = {"Authorization": f"Basic {login}", "Content-Type": "application/json"}
    experiment = "skysat-llm"  # EXPERIMENT in llm/train_lora.py, which cannot be imported here (it needs pandas)
    try:
        req = urllib.request.Request(f"{uri}/api/2.0/mlflow/experiments/get-by-name?experiment_name={experiment}",
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            experiment_id = json.load(resp)["experiment"]["experiment_id"]
        query = {"experiment_ids": [experiment_id], "filter": f"attributes.start_time > {since_ms}",
                 "max_results": 1, "order_by": ["attributes.start_time DESC"]}
        req = urllib.request.Request(f"{uri}/api/2.0/mlflow/runs/search", data=json.dumps(query).encode(),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=20) as resp:
            runs = json.load(resp).get("runs", [])
    except Exception as err:  # the network, or an answer of another shape: never ends the wait
        _once(print, f"MLflow: could not look for the run ({type(err).__name__}: {err})")
        return None
    return f"{uri}/#/experiments/{experiment_id}/runs/{runs[0]['info']['run_id']}" if runs else None


def delete(pod_id: str, log=print) -> bool:
    """Deletes the pod; a pod that is already gone counts as deleted. Three tries, then a loud line."""
    for _ in range(3):
        try:
            request("DELETE", f"/pods/{pod_id}", timeout=30)
            log(f"Pod {pod_id} deleted")
            return True
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return True
            log(f"Deleting pod {pod_id}: {err.code}")
        except (*NETWORK, ValueError) as err:  # ValueError: an answer that is not JSON
            log(f"Deleting pod {pod_id}: {err}")
        time.sleep(5)
    log(f"POD DELETE FAILED: delete pod {pod_id} by hand in the RunPod console")
    return False


def outcome(lines) -> dict:
    """What the pod's log said: the trainer's exit code and its Result line (None if the log never showed them)."""
    out = {"exit_code": None, "registered": None, "version": None, "promoted": None}
    for line in lines:
        if m := re.search(r"trainer exit code (\d+)", line):
            out["exit_code"] = int(m.group(1))
        if m := re.search(r"Result: registered=(\w+) version=(\S+) promoted=(\w+)", line):
            out |= {"registered": m.group(1) == "True", "version": None if m.group(2) == "None" else m.group(2),
                    "promoted": m.group(3) == "True"}
    return out


def follow(pod_id: str, max_minutes: int, log=print, since_ms: int | None = None) -> dict:
    """Prints the pod's container log until the pod is gone. Deletes the pod once max_minutes have passed, and
    whenever this function is left while the pod may still be there (Ctrl-C, a cancelled workflow, any error).
    With since_ms (when the pod was created, in ms) it also looks for the trainer's MLflow run until it finds it, and
    comments its link once. Returns outcome() of the log."""
    deadline = time.time() + max_minutes * 60
    seen, gone, announced = set(), False, since_ms is None
    try:
        while True:
            try:
                status = request("GET", f"/pods/{pod_id}", timeout=30).get("status")
                if status in DEAD:  # still listed, still billed, and it will print nothing more
                    log(f"Pod {pod_id} is {status}")
                    break
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    log(f"Pod {pod_id} is gone")
                    gone = True
                    break
                log(f"Reading the pod: {err.code}")
            except (*NETWORK, ValueError) as err:
                log(f"Reading the pod: {err}")  # the network blinked: the deadline below still holds
            if time.time() > deadline:
                log(f"{max_minutes} minutes passed")
                break
            if not announced and (url := mlflow_new_run(since_ms)):
                comment(f"The MLflow run started: {url}", log)
                announced = True
            _print_logs(pod_id, seen, log)
            time.sleep(20)
    finally:
        if not gone:
            delete(pod_id, log)
    return outcome(line for _, line in sorted(seen, key=lambda mark: str(mark[0])))  # in time order


def server_state() -> dict | None:
    """What the server says right now: reviews marked trained and the serving model's version. None if it cannot
    be read (no QUEUE_URL here, or the server does not answer)."""
    base, state = os.environ.get("QUEUE_URL", "").rstrip("/"), {}
    login = base64.b64encode(f"{os.environ.get('QUEUE_USER')}:{os.environ.get('QUEUE_PASSWORD')}".encode()).decode()
    for key, path, field in (("trained", "/queue/stats", "trained"), ("model", "/model", "version")):
        try:
            req = urllib.request.Request(f"{base}{path}", headers={"Authorization": f"Basic {login}"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                state[key] = json.load(resp).get(field)
        except urllib.error.HTTPError as err:
            if key == "trained" or err.code != 503:  # 503 on /model: no model loaded yet
                return None
            state[key] = None
        except (*NETWORK, ValueError):
            return None
    return state


def outcome_from_server(before: dict | None, log=print) -> dict:
    """When the pod's log could not be read (the API key may not be allowed to): what the server says happened.
    A batch is marked trained only if a version was registered, and the API loads a new champion within a minute.
    Without a registration the exit code stays unknown: a failed gate, too few reviews and a crash look the same.
    It reads counts, not this run's own result: a trial with --keep-pending registers without marking the batch
    trained (reported as not registered), and another trainer finishing in the same minutes would be counted."""
    out = {"exit_code": None, "registered": None, "version": None, "promoted": None}
    after = server_state()
    if not before or not after or before["trained"] is None or after["trained"] is None:
        log("The pod's log could not be read and neither could the server: the outcome is unknown")
        return out
    out["registered"] = after["trained"] > before["trained"]
    if out["registered"] and after["model"] == before["model"]:
        time.sleep(75)  # the API looks for a new champion every 60 s
        after = server_state() or after
    out["promoted"] = out["registered"] and after["model"] != before["model"]
    if out["registered"]:
        out |= {"exit_code": 0, "version": after["model"] if out["promoted"] else None}
    log(f"The pod's log could not be read. From the server: reviews marked trained {before['trained']} -> "
        f"{after['trained']}, serving model {before['model']} -> {after['model']}"
        + ("" if out["registered"] else ": nothing was registered (a failed gate, too few reviews, or a crash)"))
    return out


def sweep(log=print) -> int:
    """Deletes every pod named like a trainer pod that is still there. Returns how many it found: after a training
    run the answer must be 0, so a workflow can fail loudly on anything else."""
    listed = request("GET", "/pods")
    listed = listed if isinstance(listed, list) else listed.get("pods", [])
    pods = [p for p in listed if str(p.get("name", "")).startswith(POD_NAME)]
    for p in pods:
        log(f"Trainer pod {p.get('id')} was still there")
        delete(p["id"], log)
    return len(pods)


_told: set = set()


def _once(log, text: str) -> None:
    """Says a thing about the log stream once, however often it happens."""
    if text not in _told:
        _told.add(text)
        log(text)


def _print_logs(pod_id: str, seen: set, log) -> None:
    """The log is a Server-Sent Events stream: read for a few seconds, print lines not printed yet. A problem with
    the stream is said once and never ends the wait: without the log the run counts as failed, so it must be visible."""
    key = os.environ["RUNPOD_API_KEY"]
    req = urllib.request.Request(f"{API}/pods/{pod_id}/logs?source=container&tail=1000",
                                 headers={"Authorization": f"Bearer {key}", "Accept": "text/event-stream"})
    started, got = time.time(), 0
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            for raw in resp:
                line = raw.decode(errors="replace").strip()
                got += bool(line)
                if line.startswith("data:"):
                    try:
                        event = json.loads(line[5:])
                    except ValueError:
                        _once(log, f"Pod log: a line that is not JSON: {line[:200]}")
                        continue
                    if not isinstance(event, dict) or "line" not in event:
                        _once(log, f"Pod log: an event without a line: {line[:200]}")
                        continue
                    mark = (event.get("ts"), event.get("line"))
                    if mark not in seen:
                        seen.add(mark)
                        log(f"  pod | {event.get('line', '')}")
                elif line and not line.startswith((":", "event:", "id:", "retry:")):
                    _once(log, f"Pod log: an unexpected line: {line[:200]}")
                if time.time() - started > 8:
                    break
    except urllib.error.HTTPError as err:
        _once(log, f"Pod log: HTTP {err.code} {err.read().decode(errors='replace').strip()[:200]}")
    except NETWORK as err:
        if not got:  # nothing at all: no log yet (the image is still being pulled), or the stream does not answer
            _once(log, f"Pod log: nothing received ({type(err).__name__}: {err})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--branch", default=os.environ.get("TRAIN_BRANCH", "main"), help="the pushed branch the pod runs")
    p.add_argument("--trainer-args", default=os.environ.get("TRAINER_ARGS", ""), help="passed to llm.trainer")
    p.add_argument("--gpu", action="append", help="GPU type id to try (repeat for a list); default: cheapest first")
    p.add_argument("--max-minutes", type=int, default=90)
    p.add_argument("--wait", action="store_true", help="follow the log; delete the pod when it is done or too old")
    p.add_argument("--dry-run", action="store_true", help="print the request for the first GPU, start nothing")
    p.add_argument("--sweep", action="store_true",
                   help="start nothing: delete trainer pods that are still there; exit code 1 if there were any")
    args = p.parse_args()
    if args.sweep:
        left = sweep()
        print(f"Trainer pods left behind: {left}")
        return 1 if left else 0
    if args.dry_run:
        body = pod_body((args.gpu or GPUS)[0], args.branch, args.trainer_args, args.max_minutes)
        body["env"] = {k: ("***" if "PASSWORD" in k else v) for k, v in body["env"].items()}
        print(json.dumps(body, indent=2))
        return 0
    before = server_state() if args.wait else None
    started = time.time()
    pod = launch(args.gpu, args.branch, args.trainer_args, args.max_minutes)
    if os.environ.get("GITHUB_OUTPUT"):  # at once: the workflow's cooldown counts runs that rented a pod
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"pod_id={pod.get('id')}\n")
    if not args.wait:
        print(f"Not waiting. Delete it by hand if needed: curl -X DELETE -H 'Authorization: Bearer $RUNPOD_API_KEY' "
              f"{API}/pods/{pod['id']}")
        return 0
    result = follow(pod["id"], args.max_minutes + 5, since_ms=int(started * 1000))
    minutes, seconds = divmod(int(time.time() - started), 60)
    comment(f"The pod is gone: pod {pod['id']} existed {minutes} min {seconds} s.")
    if result["exit_code"] is None:
        result = outcome_from_server(before)
    print(f"Outcome: {result}")
    if os.environ.get("GITHUB_OUTPUT"):  # for .github/workflows/train.yml
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write("".join(f"{k}={'' if v is None else str(v).lower()}\n" for k, v in result.items()))
    return 0 if result["exit_code"] == 0 else 1  # an exit code the log never showed counts as a failure


if __name__ == "__main__":
    sys.exit(main())

"""Starts the trainer (llm/trainer.py) on the cheapest RunPod GPU that is free, and makes sure the pod goes away.

    RUNPOD_API_KEY=... python -m llm.runpod_pod --wait       # start, follow the log, delete the pod at the end
    python -m llm.runpod_pod --dry-run                       # print the request, start nothing

The pod clones this repo's --branch from GitHub (the code must be pushed), installs transformers/peft/mlflow on the
image's own PyTorch, downloads the base model, runs the trainer against QUEUE_URL and deletes itself when the trainer
exits. Two more guards against a forgotten pod: a timer inside the pod deletes it after --max-minutes, and with
--wait this script deletes it too once --max-minutes have passed.

GPUs are tried cheapest first (community cloud prices read from RunPod's catalog on 2026-11-16, $/hour):
RTX 3070 8 GB 0.13, RTX A4000 16 GB 0.17, RTX 3080 Ti 12 GB 0.18, RTX A4500 20 GB 0.19, RTX 4000 Ada 20 GB 0.20,
RTX 3090 24 GB 0.22. A 1.5B model with LoRA and gradient checkpointing fits 8 GB (measured on an RTX 4070 8 GB).

Env passed to the pod (from this process's env): QUEUE_URL, QUEUE_USER, QUEUE_PASSWORD (the gateway's `trainer`
login), MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD (DagsHub). They are visible in the pod's settings on RunPod.
RunPod gives every pod RUNPOD_POD_ID and a key scoped to that pod, which the pod uses to delete itself.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://api.runpod.io/v2"
IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
GPUS = ["NVIDIA GeForce RTX 3070", "NVIDIA RTX A4000", "NVIDIA GeForce RTX 3080 Ti", "NVIDIA RTX A4500",
        "NVIDIA RTX 4000 Ada Generation", "NVIDIA GeForce RTX 3090"]
REPO = "https://github.com/deensza/skysat-mlops.git"
PASS_ENV = ["QUEUE_URL", "QUEUE_USER", "QUEUE_PASSWORD",
            "MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]
PACKAGES = "transformers==5.20.2 peft==0.21.2 mlflow-skinny==3.17.1 pandas==2.2.3"

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


def pod_body(gpu: str, branch: str, trainer_args: str, max_minutes: int, name: str = "skysat-trainer") -> dict:
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
            log(f"No {gpu}: {errors[-1]}")
            continue
        log(f"Pod {pod.get('id')} created on {gpu} (branch {branch}, trainer args '{trainer_args}')")
        return pod
    raise RuntimeError("No GPU had capacity:\n" + "\n".join(errors))


def follow(pod_id: str, max_minutes: int, log=print) -> None:
    """Prints the pod's container log until the pod is gone; deletes it once max_minutes have passed."""
    deadline = time.time() + max_minutes * 60
    seen = set()
    while True:
        try:
            request("GET", f"/pods/{pod_id}", timeout=30)
        except urllib.error.HTTPError as err:
            if err.code == 404:
                log(f"Pod {pod_id} is gone")
                return
            log(f"Reading the pod: {err.code}")
        if time.time() > deadline:
            log(f"{max_minutes} minutes passed: deleting pod {pod_id}")
            request("DELETE", f"/pods/{pod_id}")
            return
        _print_logs(pod_id, seen, log)
        time.sleep(20)


def _print_logs(pod_id: str, seen: set, log) -> None:
    """The log is a Server-Sent Events stream: read for a few seconds, print lines not printed yet."""
    key = os.environ["RUNPOD_API_KEY"]
    req = urllib.request.Request(f"{API}/pods/{pod_id}/logs?source=container",
                                 headers={"Authorization": f"Bearer {key}", "Accept": "text/event-stream"})
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            for raw in resp:
                line = raw.decode(errors="replace").strip()
                if line.startswith("data:"):
                    try:
                        event = json.loads(line[5:])
                    except ValueError:
                        continue
                    mark = (event.get("ts"), event.get("line"))
                    if mark not in seen:
                        seen.add(mark)
                        log(f"  pod | {event.get('line', '')}")
                if time.time() - started > 8:
                    break
    except (urllib.error.URLError, TimeoutError, OSError):
        pass  # no log yet (the image is still being pulled), or the stream timed out: try again


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--branch", default=os.environ.get("TRAIN_BRANCH", "main"), help="the pushed branch the pod runs")
    p.add_argument("--trainer-args", default=os.environ.get("TRAINER_ARGS", ""), help="passed to llm.trainer")
    p.add_argument("--gpu", action="append", help="GPU type id to try (repeat for a list); default: cheapest first")
    p.add_argument("--max-minutes", type=int, default=90)
    p.add_argument("--wait", action="store_true", help="follow the log; delete the pod when it is done or too old")
    p.add_argument("--dry-run", action="store_true", help="print the request for the first GPU, start nothing")
    args = p.parse_args()
    if args.dry_run:
        body = pod_body((args.gpu or GPUS)[0], args.branch, args.trainer_args, args.max_minutes)
        body["env"] = {k: ("***" if "PASSWORD" in k else v) for k, v in body["env"].items()}
        print(json.dumps(body, indent=2))
        return 0
    pod = launch(args.gpu, args.branch, args.trainer_args, args.max_minutes)
    if args.wait:
        follow(pod["id"], args.max_minutes + 5)
    else:
        print(f"Not waiting. Delete it by hand if needed: curl -X DELETE -H 'Authorization: Bearer $RUNPOD_API_KEY' "
              f"{API}/pods/{pod['id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

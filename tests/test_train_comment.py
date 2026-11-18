""".github/scripts/train-comment.sh run with a fake gh first on PATH, and the hooks in llm/runpod_pod.py that call it.
No GitHub, no RunPod, no MLflow, no network: every request is replaced."""

import io
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from llm import runpod_pod

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / ".github" / "scripts" / "train-comment.sh"
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")

# `gh issue list ...` prints the numbers in FAKE_OPEN (as --jq would), `gh issue create` prints an issue address;
# every call is written down
FAKE_GH = """#!/bin/bash
printf '%s\\n' "$*" >> "$FAKE_GH_LOG"
if [ "$1 $2" = "issue list" ]; then printf '%s' "$FAKE_OPEN"; fi
if [ "$1 $2" = "issue create" ]; then echo "https://github.com/x/y/issues/57"; fi
if [ "$1 $2" = "issue comment" ] && [ -n "$FAKE_COMMENT_FAILS" ]; then echo "HTTP 403" >&2; exit 1; fi
exit 0
"""
RUN = {"GITHUB_SERVER_URL": "https://github.com", "GITHUB_REPOSITORY": "x/y", "GITHUB_RUN_ID": "999"}


def run(tmp_path, text="GPU rented", open_issues="", **env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "gh").write_text(FAKE_GH)
    (bin_dir / "gh").chmod(0o755)
    log = tmp_path / "gh.log"
    log.write_text("")  # the calls of this run only
    done = subprocess.run(
        ["bash", str(SCRIPT), text], capture_output=True, text=True, timeout=30,
        env={"PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_GH_LOG": str(log), "FAKE_OPEN": open_issues,
             "TRAIN_ISSUE_FILE": str(tmp_path / "issue"), **RUN, **env})
    return done, log.read_text().splitlines()


@needs_bash
def test_no_open_issue_creates_it_once_and_comments_with_the_run_link(tmp_path):
    done, calls = run(tmp_path, "A GPU was rented: pod abc")
    assert done.returncode == 0, done.stderr
    assert calls[0].startswith("issue list --state open") and "--search" not in calls[0]
    assert calls[1].startswith('issue create --title Training runs --body ')
    assert calls[2].startswith("issue comment 57 --body ")  # the number after the last / of the created address
    assert "A GPU was rented: pod abc" in calls[2]
    assert "([run 999](https://github.com/x/y/actions/runs/999))" in calls[2]
    assert len(calls) == 3


@needs_bash
def test_an_open_issue_is_used_and_no_second_one_is_created(tmp_path):
    done, calls = run(tmp_path, open_issues="31\n40\n")
    assert done.returncode == 0
    assert not any(c.startswith("issue create") for c in calls)
    assert calls[-1].startswith("issue comment 31 --body ")


@needs_bash
def test_a_second_call_of_the_same_run_does_not_list_again(tmp_path):
    run(tmp_path, "first")
    done, calls = run(tmp_path, "second")  # same TRAIN_ISSUE_FILE: the number is already known
    assert done.returncode == 0
    assert calls == [calls[0]] and calls[0].startswith("issue comment 57 --body ") and "second" in calls[0]


@needs_bash
def test_a_failed_comment_gives_a_nonzero_exit(tmp_path):
    done, _ = run(tmp_path, open_issues="31\n", FAKE_COMMENT_FAILS="1")
    assert done.returncode != 0


@needs_bash
def test_a_dry_run_creates_and_comments_nothing(tmp_path):
    done, calls = run(tmp_path, "hello", DRY_RUN="1")
    assert done.returncode == 0 and "Would comment" in done.stdout and "hello" in done.stdout
    assert all(c.startswith("issue list") for c in calls)
    done, calls = run(tmp_path, "hello", open_issues="31\n", DRY_RUN="1")
    assert "issue #31" in done.stdout and all(c.startswith("issue list") for c in calls)


@needs_bash
def test_without_a_run_id_there_is_no_link(tmp_path):
    done, calls = run(tmp_path, "hello", open_issues="31\n", GITHUB_RUN_ID="")
    assert done.returncode == 0 and calls[-1].endswith("--body hello")


def fake_script(tmp_path, exit_code=0):
    script = tmp_path / "fake-comment.sh"
    script.write_text(f'printf "%s" "$1" > "{tmp_path}/said"\necho "boom" >&2\nexit {exit_code}\n')
    return script


@needs_bash
def test_comment_passes_the_text_to_the_script(tmp_path, monkeypatch):
    monkeypatch.setenv("COMMENT_SCRIPT", str(fake_script(tmp_path)))
    runpod_pod.comment("hello there")
    assert (tmp_path / "said").read_text() == "hello there"


@needs_bash
def test_comment_never_raises_and_says_why(tmp_path, monkeypatch):
    lines = []
    monkeypatch.setenv("COMMENT_SCRIPT", str(fake_script(tmp_path, exit_code=1)))
    runpod_pod.comment("hello", lines.append)
    assert len(lines) == 1 and lines[0].startswith("Comment not posted") and "boom" in lines[0]
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no bash")))
    runpod_pod.comment("hello", lines.append)
    assert len(lines) == 2 and "no bash" in lines[1]


def test_comment_without_a_script_runs_nothing(monkeypatch):
    monkeypatch.delenv("COMMENT_SCRIPT", raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("a script was run"))
    runpod_pod.comment("hello")


def test_launch_comments_the_pod_the_gpu_and_the_price(monkeypatch):
    monkeypatch.setattr(runpod_pod, "request", lambda *a, **k: {"id": "pod1", "cost": 0.13})
    for k in ("QUEUE_URL", "QUEUE_USER", "QUEUE_PASSWORD", "RUNPOD_API_KEY"):
        monkeypatch.setenv(k, "dummy")
    monkeypatch.setenv("COMMENT_SCRIPT", "script.sh")
    said = []
    monkeypatch.setattr(runpod_pod, "comment", lambda text, log=print: said.append(text))
    runpod_pod.launch(log=lambda line: None)
    assert len(said) == 1 and "pod1" in said[0] and "NVIDIA GeForce RTX 3070" in said[0] and "0.13" in said[0]
    assert "dummy" not in said[0]  # no env value goes into a comment
    monkeypatch.setattr(runpod_pod, "request", lambda *a, **k: {"id": "pod2"})  # no cost field: the list price
    runpod_pod.launch(log=lambda line: None)
    assert "list price of 2026-11-16" in said[1] and "0.13" in said[1]


class Answer(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_mlflow(monkeypatch, runs, seen=None):
    def urlopen(req, timeout=None):
        if seen is not None:
            seen.append((req.full_url, req.data, req.get_header("Authorization")))
        if "get-by-name" in req.full_url:
            return Answer(json.dumps({"experiment": {"experiment_id": "7"}}).encode())
        answer = {"runs": [{"info": {"run_id": r}} for r in runs]} if runs else {}
        return Answer(json.dumps(answer).encode())
    monkeypatch.setattr("urllib.request.urlopen", urlopen)


def test_mlflow_new_run_returns_the_link_of_the_newest_run(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test/")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "u")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "p")
    seen = []
    fake_mlflow(monkeypatch, ["abc123"], seen)
    assert runpod_pod.mlflow_new_run(1234) == "https://mlflow.example.test/#/experiments/7/runs/abc123"
    assert "experiment_name=skysat-llm" in seen[0][0] and seen[0][2].startswith("Basic ")
    assert json.loads(seen[1][1])["filter"] == "attributes.start_time > 1234"


def test_mlflow_new_run_is_none_without_a_run_or_a_server(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    fake_mlflow(monkeypatch, [])
    assert runpod_pod.mlflow_new_run(1) is None  # the answer is {}
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("asked a file store"))
    assert runpod_pod.mlflow_new_run(1) is None
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    assert runpod_pod.mlflow_new_run(1) is None


def test_the_workflow_sets_the_script_and_comments_the_ending():
    train = (ROOT / ".github" / "workflows" / "train.yml").read_text()
    assert "COMMENT_SCRIPT: .github/scripts/train-comment.sh" in train
    assert "- name: Comment the ending" in train and "steps.summary.outputs.ending" in train
    step = train.split("- name: Comment the ending")[1]
    assert "continue-on-error: true" in step and "timeout-minutes: 2" in step

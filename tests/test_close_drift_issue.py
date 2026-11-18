""".github/scripts/close-drift-issue.sh run for real with a fake gh first on PATH. No GitHub, no network.
The same script closed the real issue of 18 Nov 2026 (deensza/skysat-mlops#24) when it was run by hand."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / ".github" / "scripts" / "close-drift-issue.sh"
TITLE = "Drift detected: the serving model fell below the gate"

# `gh issue list ...` prints the numbers in FAKE_OPEN (one per line, as --jq would); every call is written down
FAKE_GH = """#!/bin/bash
printf '%s\\n' "$*" >> "$FAKE_GH_LOG"
if [ "$1 $2" = "issue list" ]; then printf '%s' "$FAKE_OPEN"; fi
if [ "$1 $2" = "issue close" ] && [ -n "$FAKE_CLOSE_FAILS" ]; then exit 1; fi
exit 0
"""


def run(tmp_path, *args, open_issues="", **env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "gh").write_text(FAKE_GH)
    (bin_dir / "gh").chmod(0o755)
    log = tmp_path / "gh.log"
    log.write_text("")
    done = subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, timeout=30,
        env={"PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_GH_LOG": str(log), "FAKE_OPEN": open_issues,
             "FAKE_CLOSE_FAILS": "", **env})
    return done, log.read_text().splitlines()


def test_a_promotion_closes_the_open_drift_issue_and_says_why(tmp_path):
    done, calls = run(tmp_path, "8", "https://example.test/runs/1", open_issues="24\n")
    assert done.returncode == 0 and "Closed issue #24" in done.stdout
    assert calls[0].startswith("issue list --state open") and TITLE in calls[0]
    assert calls[1].startswith("issue close 24 --comment ")
    assert "version 8" in calls[1] and "https://example.test/runs/1" in calls[1]
    assert len(calls) == 2


def test_no_open_issue_is_not_an_error(tmp_path):
    done, calls = run(tmp_path, "8")
    assert done.returncode == 0 and "nothing to close" in done.stdout
    assert len(calls) == 1 and calls[0].startswith("issue list")  # nothing was closed


def test_a_dry_run_closes_nothing(tmp_path):
    done, calls = run(tmp_path, "8", open_issues="24\n", DRY_RUN="1")
    assert done.returncode == 0 and "Would close issue #24" in done.stdout
    assert not any(call.startswith("issue close") for call in calls)


def test_it_needs_the_version_and_reports_a_failed_close(tmp_path):
    done, calls = run(tmp_path, open_issues="24\n")
    assert done.returncode != 0 and calls == []  # no version: gh is never called
    done, _ = run(tmp_path, "8", open_issues="24\n", FAKE_CLOSE_FAILS="1")
    assert done.returncode != 0 and "Closed issue" not in done.stdout  # the workflow step shows the failure


def test_the_title_is_the_one_the_drift_check_opens_and_train_calls_the_script():
    assert f'title="{TITLE}"' in (ROOT / ".github" / "workflows" / "drift.yml").read_text()
    assert f'title="{TITLE}"' in SCRIPT.read_text()
    train = (ROOT / ".github" / "workflows" / "train.yml").read_text()
    assert "close-drift-issue.sh" in train and "issues: write" in train
    assert "if: steps.train.outputs.promoted == 'true'" in train  # only a promotion answers the alarm

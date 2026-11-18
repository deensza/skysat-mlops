"""monitoring/drift-trigger.sh run for real with a fake curl first on PATH. No GitHub, no network.
The fake curl writes down its arguments and what it was given on stdin, and prints the HTTP code in FAKE_CODE."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "monitoring" / "drift-trigger.sh"
TOKEN = "ghp_TESTTOKEN0123456789"
URL = "https://api.github.com/repos/deensza/skysat-mlops/actions/workflows/drift.yml/dispatches"

FAKE_CURL = """#!/bin/bash
printf '%s\\n' "$@" > "$FAKE_ARGV"
cat > "$FAKE_STDIN"
printf '%s' "${FAKE_CODE:-204}"
"""


def run(tmp_path, token=TOKEN, **env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "curl").write_text(FAKE_CURL)
    (bin_dir / "curl").chmod(0o755)
    token_file = tmp_path / "token"
    if token is not None:
        token_file.write_text(token)
    argv, stdin = tmp_path / "argv", tmp_path / "stdin"
    done = subprocess.run(
        ["bash", str(SCRIPT)], capture_output=True, text=True, timeout=30,
        env={"PATH": f"{bin_dir}:{os.environ['PATH']}", "DRIFT_TRIGGER_TOKEN_FILE": str(token_file),
             "FAKE_ARGV": str(argv), "FAKE_STDIN": str(stdin), **env})
    return done, (argv.read_text().splitlines() if argv.exists() else None), (stdin.read_text() if stdin.exists() else None)


def test_a_204_is_logged_ok_and_the_token_only_goes_through_stdin(tmp_path):
    done, argv, stdin = run(tmp_path, token=TOKEN + "\n")
    assert done.returncode == 0
    assert len(done.stdout.splitlines()) == 1 and "dispatch drift.yml: HTTP 204" in done.stdout and "(ok)" in done.stdout
    assert URL in argv and "--max-time" in argv and '{"ref":"main"}' in argv
    assert argv[argv.index("-X") + 1] == "POST"
    assert "Accept: application/vnd.github+json" in argv
    assert not any(TOKEN in part for part in argv)
    assert TOKEN not in done.stdout and TOKEN not in done.stderr
    assert f'Authorization: Bearer {TOKEN}"' in stdin  # the newline of the file is gone


def test_a_403_is_logged_failed_and_exits_1(tmp_path):
    done, argv, stdin = run(tmp_path, FAKE_CODE="403")
    assert done.returncode == 1
    assert len(done.stdout.splitlines()) == 1 and "HTTP 403" in done.stdout and "(FAILED)" in done.stdout
    assert TOKEN not in done.stdout + done.stderr


@pytest.mark.parametrize("token", [None, "", "\n"], ids=["missing file", "empty file", "only a newline"])
def test_no_token_means_no_call(tmp_path, token):
    done, argv, stdin = run(tmp_path, token=token)
    assert done.returncode == 1
    assert argv is None  # curl was never called
    assert len(done.stdout.splitlines()) == 1 and "no token" in done.stdout and "(FAILED)" in done.stdout

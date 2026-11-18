"""monitoring/watchdog.sh run for real with a fake docker and a fake curl first on PATH. No Docker, no network."""

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")

SCRIPT = Path(__file__).resolve().parent.parent / "monitoring" / "watchdog.sh"

FAKE_DOCKER = """#!/bin/bash
echo "$*" >> "$FAKE_DOCKER_LOG"
if [ "$1" = ps ]; then
  case "$*" in
    *status=exited*) echo -n "$FAKE_STOPPED" ;;
    *service=api*) echo api123 ;;
  esac
fi
"""

FAKE_CURL = """#!/bin/bash
echo -n "$FAKE_HTTP_CODE"
[ "$FAKE_HTTP_CODE" = 000 ] && exit 28
exit 0
"""


@pytest.fixture
def box(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("docker", FAKE_DOCKER), ("curl", FAKE_CURL)):
        (bin_dir / name).write_text(textwrap.dedent(body))
        (bin_dir / name).chmod(0o755)
    return tmp_path


def run(box, code, stopped=""):
    env = {
        "PATH": f"{box / 'bin'}:/usr/bin:/bin",
        "WATCHDOG_STATE": str(box / "fails"),
        "WATCHDOG_OFF": str(box / "off"),
        "FAKE_DOCKER_LOG": str(box / "docker.log"),
        "FAKE_HTTP_CODE": code,
        "FAKE_STOPPED": stopped,
    }
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, check=True).stdout


def docker_calls(box):
    log = box / "docker.log"
    return log.read_text().splitlines() if log.exists() else []


def state(box):
    return (box / "fails").read_text().strip()


def test_healthy_resets_counter(box):
    (box / "fails").write_text("2\n")
    run(box, "200")
    assert state(box) == "0"
    assert not any(c.startswith("restart") for c in docker_calls(box))


def test_503_is_not_a_failure(box):
    (box / "fails").write_text("2\n")
    for _ in range(3):
        out = run(box, "503")
        assert "no model yet" in out
        assert "failed" not in out
    assert not any(c.startswith("restart") for c in docker_calls(box))
    assert state(box) == "0"


def test_no_answer_restarts_api_on_third_run(box):
    run(box, "000")
    assert state(box) == "1"
    run(box, "000")
    assert state(box) == "2"
    assert not any(c.startswith("restart") for c in docker_calls(box))
    out = run(box, "000")
    assert "health check failed (3, status 000)" in out
    assert "restart api123" in docker_calls(box)
    assert state(box) == "0"


def test_success_after_failures_resets_counter(box):
    run(box, "000")
    run(box, "502")
    assert state(box) == "2"
    run(box, "200")
    assert state(box) == "0"
    assert not any(c.startswith("restart") for c in docker_calls(box))


def test_pause_file_touches_nothing(box):
    (box / "off").touch()
    (box / "fails").write_text("1\n")
    out = run(box, "000", stopped="web9")
    assert docker_calls(box) == []
    assert state(box) == "1"
    assert "paused" in out


def test_stopped_container_is_started(box):
    run(box, "200", stopped="web9")
    assert "start web9" in docker_calls(box)

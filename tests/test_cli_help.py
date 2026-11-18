"""Every command of the project prints its help. argparse formats help texts with %, so a bare % in one of them
crashed `python -m llm.trainer --help` until 18 Nov 2026 (the training itself was not affected)."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("module", ["llm.trainer", "llm.runpod_pod", "llm.drift_check", "llm.promote",
                                    "simulator.replay"])
def test_help_is_printed(module):
    done = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert done.returncode == 0, done.stderr[-400:]
    assert "usage:" in done.stdout

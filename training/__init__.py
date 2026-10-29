"""Training code. macOS only: PyTorch brings its own OpenMP runtime and LightGBM / xgboost use Homebrew's
libomp; with both running multi-threaded in one process (`make test`, `make ct-local ARGS=--all`) Python
crashes with a segmentation fault. One OpenMP thread avoids it.

Every platform: PyTorch's OpenMP threads busy-wait after use, and LightGBM's threads (a second OpenMP runtime,
also on Linux) then fight them: LightGBM on the full data took 271 s instead of 6 s when PyTorch was loaded first.
OMP_WAIT_POLICY=PASSIVE lets idle threads sleep (MLP speed unchanged). Must be set before PyTorch is imported."""

import importlib.util
import os
import sys

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
if sys.platform == "darwin" and any(importlib.util.find_spec(m) for m in ("lightgbm", "xgboost")):
    os.environ.setdefault("OMP_NUM_THREADS", "1")

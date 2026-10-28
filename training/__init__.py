"""Training code. macOS only: PyTorch brings its own OpenMP runtime and LightGBM / xgboost use Homebrew's
libomp; with both running multi-threaded in one process (`make test`, `make ct-local ARGS=--all`) Python
crashes with a segmentation fault. One OpenMP thread avoids it. Linux (CI, GitHub CT, the Docker API) shares
one runtime and is not affected."""

import importlib.util
import os
import sys

if sys.platform == "darwin" and any(importlib.util.find_spec(m) for m in ("lightgbm", "xgboost")):
    os.environ.setdefault("OMP_NUM_THREADS", "1")

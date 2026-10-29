"""The tests import PyTorch before `training`, so the OpenMP setting from training/__init__.py comes too late
there; set it here (pytest loads this file first)."""

import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

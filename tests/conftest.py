"""Set OMP_WAIT_POLICY before any test imports PyTorch (pytest loads this file first)."""

import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

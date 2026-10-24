"""Filesystem locations that resolve both locally and inside a Kaggle notebook."""

import os
from pathlib import Path

COMPETITION = "name-that-llm"

# Set in every Kaggle kernel (interactive and the hidden-test rerun).
ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

REPO_ROOT = Path(__file__).resolve().parents[2]

if ON_KAGGLE:
    DATA_DIR = Path("/kaggle/input/competitions") / COMPETITION
    OUTPUT_DIR = Path("/kaggle/working")
else:
    DATA_DIR = REPO_ROOT / "data" / "raw"
    OUTPUT_DIR = REPO_ROOT / "outputs"

# Escape hatch for unusual mounts or alternate copies of the data.
DATA_DIR = Path(os.environ.get("NTL_DATA_DIR", DATA_DIR))

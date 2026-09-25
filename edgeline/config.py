"""Paths and settings. Everything lives under the repo's data/ directory by default."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("EDGELINE_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.environ.get("EDGELINE_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
DB_PATH = Path(os.environ.get("EDGELINE_DB", DATA_DIR / "edgeline.db"))
ARTIFACT_DIR = Path(os.environ.get("EDGELINE_ARTIFACTS", ROOT / "models" / "artifacts"))

USER_AGENT = os.environ.get(
    "EDGELINE_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
)

# Default bettable thresholds (LCSLarry's shipped defaults for DFS books).
DEFAULT_MIN_PROB = 0.60
DEFAULT_MIN_EV = 0.05
DEFAULT_MAX_LINE_MOVE = 0.10  # skip lines that moved more than 10% from open

for _p in (DATA_DIR, RAW_DIR, ARTIFACT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

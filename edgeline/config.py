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
# Market prior: weight given to the book's line when forming each component's mean (0 = pure model).
# Books are sharper than a short-history model; 0.25 keeps most of the model's view while tempering
# its largest disagreements. Tune once graded results accumulate.
DEFAULT_MARKET_SHRINK = 0.25
# Markets whose evidence does not support the 60% threshold: priced and graded, but not flagged bettable unless
# `--include-unproven` is passed. Empty since 2026-09-26: CS2 kills left when twelve months of history moved its
# walk-forward from 58.5% to 64.3%, COD when a second season moved it from 57.5% to 68.1% (226 picks). A market
# goes back on the list when its captured-line record contradicts the backtest.
UNPROVEN_MARKETS: set[tuple[str, str]] = set()

for _p in (DATA_DIR, RAW_DIR, ARTIFACT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

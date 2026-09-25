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
# Markets the walk-forward backtest does not support at the 60% threshold against a fair book-like setter
# (COD 57.5%; break-even is 56.2% and the interval includes losing money). They are still priced and graded,
# so the record keeps growing, but they are not flagged bettable unless `--include-unproven` is passed.
# CS2 kills sat here while the six-month model backtested at 58.5%; twelve months of history moved it to
# 64.3% (interval 62.2-66.4), so it is flagged again and the captured-line record decides whether it stays.
UNPROVEN_MARKETS = {("cod", "kills"), ("cod", "deaths"), ("cod", "assists")}

for _p in (DATA_DIR, RAW_DIR, ARTIFACT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

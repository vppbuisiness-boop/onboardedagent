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
# Slips and the bettable list only show games starting today or tomorrow in the bettor's local day (the book posts
# lines days ahead; those are priced and tracked but not proposed as bets until they are inside the horizon).
LOCAL_TZ = "America/New_York"
SLIP_HORIZON_DAYS = 1
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
# 2026-09-26 16:00 UTC: CS2 kills goes back on the list. 277 settled captured lines over 29 matches ran 136-141 (49.1%);
# the model's leans add nothing over the market's own under rate (UNDER leans 54.5% vs 59.6% of all lines settling under,
# OVER leans 33.3%), and on those lines the book's number is more accurate than the projection (MAE 4.52 vs 4.77).
# CS2 headshots stays open: 133-92 (59.1%), UNDER leans 65% against a 58% blind under rate, replay 57.7%.
UNPROVEN_MARKETS: set[tuple[str, str]] = {("cs2", "kills")}

for _p in (DATA_DIR, RAW_DIR, ARTIFACT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

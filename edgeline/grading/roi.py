"""The ROI gauge in the original product's convention, with uncertainty.

LCSLarry's headline "~29-30% ROI" is the 4-pick POWER parlay ROI implied by a 60% per-leg hit rate
on graded opening lines: 0.6^4 * 10 - 1 = 29.6%. So the target is a per-leg hit rate, and the
honest question is whether the graded sample is large enough to tell 60% from the 56.2% break-even.
This module reports the hit rate with a Wilson interval, the parlay ROI at the point estimate and at
the interval bounds, and the sample size needed for the lower bound to clear break-even at the
current rate.
"""
from __future__ import annotations

from math import sqrt

import pandas as pd

from ..ev.payouts import ladder, leg_decimal_odds

Z95 = 1.959964


def wilson(hits: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, centre - half, centre + half


def parlay_roi(hit_rate: float, book: str = "prizepicks", size: int = 4) -> float:
    payout = ladder(book, "POWER", size)[-1] + 1.0
    return hit_rate**size * payout - 1.0


def n_needed(hit_rate: float, breakeven: float, z: float = Z95) -> int | None:
    """Sample size at which the Wilson lower bound at `hit_rate` clears `breakeven` (None if hit_rate <= breakeven)."""
    if hit_rate <= breakeven:
        return None
    n = 10
    while n < 10**6:
        _, lo, _ = wilson(round(hit_rate * n), n, z)
        if lo > breakeven:
            return n
        n = int(n * 1.15) + 1
    return None


def gauge(df: pd.DataFrame, book: str = "prizepicks", label: str = "") -> dict:
    """df from grading.results.results_frame (has win_open / win_current)."""
    settled = df["win_open"].dropna()
    n, hits = int(len(settled)), int(settled.sum())
    p, lo, hi = wilson(hits, n)
    breakeven = 1.0 / leg_decimal_odds(book)
    out = {
        "label": label, "n": n, "hits": hits, "losses": n - hits, "hit_rate": p, "ci_low": lo, "ci_high": hi,
        "breakeven_leg": breakeven, "target_leg": 0.60,
        "parlay4_roi": parlay_roi(p, book) if n else float("nan"),
        "parlay4_roi_ci": (parlay_roi(lo, book), parlay_roi(hi, book)) if n else (float("nan"), float("nan")),
        "n_to_prove_breakeven": n_needed(p, breakeven) if n else None,
        "n_to_prove_60": n_needed(p, 0.60) if n else None,
    }
    return out


def format_gauge(g: dict) -> str:
    if not g["n"]:
        return f"{g['label']}: no settled lines yet"
    lines = [
        f"{g['label']}: {g['hits']}-{g['losses']} settled, hit rate {g['hit_rate']:.1%} (95% CI {g['ci_low']:.1%} to {g['ci_high']:.1%})",
        f"  4-pick parlay ROI at the point estimate {g['parlay4_roi']:+.1%}; across the CI {g['parlay4_roi_ci'][0]:+.1%} to {g['parlay4_roi_ci'][1]:+.1%}",
        f"  break-even leg rate {g['breakeven_leg']:.1%}; the 30% ROI target is a 60% leg rate",
    ]
    if g["n_to_prove_breakeven"]:
        lines.append(f"  at this rate, ~{g['n_to_prove_breakeven']:,} settled lines would put the CI's lower bound above break-even; ~{g['n_to_prove_60'] or 'n/a'} above 60%")
    else:
        lines.append("  the current rate is not above break-even")
    return "\n".join(lines)

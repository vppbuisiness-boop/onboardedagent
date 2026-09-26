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

import numpy as np
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


MIN_CLUSTERS = 5


def match_key(df: pd.DataFrame) -> pd.Series:
    """One key per match: the book re-lists later maps of a series under new game ids, so cluster on the unordered
    team pair plus the calendar day of the start time."""
    a, b = df["team"].fillna("?").astype(str), df["opponent"].fillna("?").astype(str)
    lo, hi = a.where(a <= b, b), b.where(a <= b, a)
    return lo + "|" + hi + "|" + df["start_time"].fillna("").astype(str).str.slice(0, 10)


def cluster_ci(wins: pd.Series, clusters: pd.Series, n_boot: int = 2000, seed: int = 7) -> tuple[int, float, float]:
    """Lines from one match win or lose together, so the Wilson interval overstates what a same-match block proves.
    Resample matches (clusters) with replacement and return (matches, 2.5th, 97.5th percentile of the hit rate)."""
    frame = pd.DataFrame({"w": wins.astype(float).values, "c": clusters.fillna("?").astype(str).values})
    per = frame.groupby("c")["w"].agg(["sum", "count"])
    m = len(per)
    if m == 0:
        return 0, float("nan"), float("nan")
    if m == 1:
        return 1, 0.0, 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n_boot, m))
    sums, counts = per["sum"].values[idx].sum(axis=1), per["count"].values[idx].sum(axis=1)
    rates = sums / counts
    return m, float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def gauge(df: pd.DataFrame, book: str = "prizepicks", label: str = "") -> dict:
    """df from grading.results.results_frame (has win_open / win_current)."""
    settled = df["win_open"].dropna()
    n, hits = int(len(settled)), int(settled.sum())
    p, lo, hi = wilson(hits, n)
    matches, clo, chi = 0, float("nan"), float("nan")
    if n and {"team", "opponent", "start_time"} <= set(df.columns):
        matches, clo, chi = cluster_ci(settled, match_key(df.loc[settled.index]))
        if matches < MIN_CLUSTERS:  # a handful of matches cannot bound the rate; say so instead of printing a narrow interval
            clo, chi = float("nan"), float("nan")
    breakeven = 1.0 / leg_decimal_odds(book)
    out = {
        "label": label, "n": n, "hits": hits, "losses": n - hits, "hit_rate": p, "ci_low": lo, "ci_high": hi,
        "breakeven_leg": breakeven, "target_leg": 0.60,
        "parlay4_roi": parlay_roi(p, book) if n else float("nan"),
        "parlay4_roi_ci": (parlay_roi(lo, book), parlay_roi(hi, book)) if n else (float("nan"), float("nan")),
        "n_to_prove_breakeven": n_needed(p, breakeven) if n else None,
        "n_to_prove_60": n_needed(p, 0.60) if n else None,
        "matches": matches, "cluster_ci_low": clo, "cluster_ci_high": chi,
    }
    return out


def format_gauge(g: dict) -> str:
    if not g["n"]:
        return f"{g['label']}: no settled lines yet"
    lines = [
        f"{g['label']}: {g['hits']}-{g['losses']} settled, hit rate {g['hit_rate']:.1%} (95% CI {g['ci_low']:.1%} to {g['ci_high']:.1%})",
        f"  4-pick parlay ROI at the point estimate {g['parlay4_roi']:+.1%}; across the CI {g['parlay4_roi_ci'][0]:+.1%} to {g['parlay4_roi_ci'][1]:+.1%}",
        (f"  across {g['matches']} matches; match-cluster 95% CI {g['cluster_ci_low']:.1%} to {g['cluster_ci_high']:.1%}"
         if g.get("matches", 0) >= MIN_CLUSTERS else
         f"  across {g.get('matches', 0)} match(es): too few independent matches to bound the rate; same-match lines win or lose together"),
        f"  break-even leg rate {g['breakeven_leg']:.1%}; the 30% ROI target is a 60% leg rate",
    ]
    if g["n_to_prove_breakeven"]:
        lines.append(f"  at this rate, ~{g['n_to_prove_breakeven']:,} settled lines would put the CI's lower bound above break-even; ~{g['n_to_prove_60'] or 'n/a'} above 60%")
    else:
        lines.append("  the current rate is not above break-even")
    return "\n".join(lines)

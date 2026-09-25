"""Expected value math for pick'em slips and straight bets.

Slip EV uses the exact Poisson-binomial distribution of the number of legs
that hit (legs may have different probabilities), evaluated against a payout
ladder. With a single common probability this reduces to the binomial
expansion LCSLarry's calculator uses.
"""
from __future__ import annotations

from math import comb
from typing import Iterable, Sequence

import numpy as np

from .payouts import ladder, leg_decimal_odds


def leg_ev(p: float, decimal_odds: float) -> float:
    """EV of a straight bet at the given decimal odds, per 1 unit staked."""
    return p * (decimal_odds - 1.0) - (1.0 - p)


def leg_ev_on_book(p: float, book: str) -> float:
    """EV of a single leg priced at the book's implied per-leg odds."""
    return leg_ev(p, leg_decimal_odds(book))


def poisson_binomial_pmf(probs: Sequence[float]) -> np.ndarray:
    """P(exactly k of n independent legs hit), k = 0..n, via DP convolution."""
    pmf = np.array([1.0])
    for p in probs:
        p = float(p)
        new = np.zeros(len(pmf) + 1)
        new[:-1] += pmf * (1.0 - p)
        new[1:] += pmf * p
        pmf = new
    return pmf


def slip_ev_from_pmf(pmf: Sequence[float], net_ladder: Sequence[float]) -> float:
    if len(pmf) != len(net_ladder):
        raise ValueError("pmf and ladder must have the same length (n legs + 1)")
    return float(np.dot(np.asarray(pmf, dtype=float), np.asarray(net_ladder, dtype=float)))


def slip_ev(probs: Sequence[float], book: str, slip_type: str) -> float:
    """EV per 1 unit of an n-leg slip with independent legs at the given probabilities."""
    probs = list(probs)
    net = ladder(book, slip_type, len(probs))
    return slip_ev_from_pmf(poisson_binomial_pmf(probs), net)


def slip_hit_prob(probs: Sequence[float], book: str, slip_type: str) -> float:
    """Probability the slip returns more than the stake (net > 0)."""
    probs = list(probs)
    net = ladder(book, slip_type, len(probs))
    pmf = poisson_binomial_pmf(probs)
    return float(sum(pmf[k] for k in range(len(net)) if net[k] > 0))


def slip_ev_common_p(p: float, net_ladder: Sequence[float]) -> float:
    """Binomial shortcut used by LCSLarry's calculator: every leg at the same p."""
    n = len(net_ladder) - 1
    return float(sum(comb(n, k) * p**k * (1 - p) ** (n - k) * net_ladder[k] for k in range(n + 1)))


def breakeven_hit_rate(net_ladder: Sequence[float], lo: float = 0.30, hi: float = 0.99) -> float:
    """Per-leg hit rate at which the slip's EV is zero (bisection)."""
    for _ in range(80):
        mid = (lo + hi) / 2
        if slip_ev_common_p(mid, net_ladder) < 0:
            lo = mid
        else:
            hi = mid
    return hi


def kelly_fraction(p: float, decimal_odds: float, fraction: float = 1.0) -> float:
    """Kelly stake as a fraction of bankroll for a straight bet; clipped at 0."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    f = (p * b - (1.0 - p)) / b
    return max(0.0, f * fraction)


def stack_probs_from_joint(joint_hit: float, p_a: float, p_b: float) -> dict:
    """Helper for correlated pairs: describe how much a joint probability exceeds independence."""
    indep = p_a * p_b
    return {"joint": joint_hit, "independent": indep, "lift": (joint_hit / indep - 1.0) if indep else float("nan")}


def ev_table(book: str, slip_type: str, size: int, hit_rates: Iterable[float]) -> list[tuple[float, float]]:
    net = ladder(book, slip_type, size)
    return [(float(h), slip_ev_common_p(float(h), net)) for h in hit_rates]

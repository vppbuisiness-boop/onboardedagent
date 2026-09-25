"""Monte Carlo bankroll simulation for fixed-payout slips.

Mirrors the simulator in LCSLarry's EV calculator: each day place `bets_per_day`
slips of `size` legs, each leg hitting with probability `hit_rate`; settle each
slip on the payout ladder; track the balance path, max drawdown and run-up.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .payouts import ladder


@dataclass
class SimResult:
    balance: np.ndarray  # cumulative units by day (len days + 1)
    total_bets: int
    bets_won: int
    prop_hits: int
    total_props: int
    max_drawdown: float
    max_runup: float

    @property
    def final(self) -> float:
        return float(self.balance[-1])

    @property
    def roi_per_bet(self) -> float:
        return self.final / self.total_bets if self.total_bets else 0.0


def _drawdown(path: np.ndarray) -> tuple[float, float]:
    peak = -np.inf
    max_dd = 0.0
    for v in path:
        peak = max(peak, v)
        max_dd = max(max_dd, peak - v)
    return float(max_dd), float(path.max())


def simulate_path(hit_rate: float, days: int, bets_per_day: int, size: int, book: str, slip_type: str,
                  rng: np.random.Generator) -> SimResult:
    net = np.asarray(ladder(book, slip_type, size), dtype=float)
    hits = rng.random((days, bets_per_day, size)) < hit_rate
    k = hits.sum(axis=2)  # legs hit per slip
    slip_net = net[k]
    daily = slip_net.sum(axis=1)
    balance = np.concatenate([[0.0], np.cumsum(daily)])
    dd, ru = _drawdown(balance)
    return SimResult(
        balance=balance,
        total_bets=days * bets_per_day,
        bets_won=int((slip_net > 0).sum()),
        prop_hits=int(hits.sum()),
        total_props=days * bets_per_day * size,
        max_drawdown=dd,
        max_runup=ru,
    )


def simulate(hit_rate: float, days: int = 30, bets_per_day: int = 4, size: int = 4, book: str = "prizepicks",
             slip_type: str = "POWER", iterations: int = 2000, seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    runs = [simulate_path(hit_rate, days, bets_per_day, size, book, slip_type, rng) for _ in range(iterations)]
    runs.sort(key=lambda r: r.final)
    p25, p50, p75 = runs[iterations // 4], runs[iterations // 2], runs[(3 * iterations) // 4]
    finals = np.array([r.final for r in runs])
    return {
        "median": p50,
        "p25": p25,
        "p75": p75,
        "mean_final": float(finals.mean()),
        "prob_profit": float((finals > 0).mean()),
        "worst_drawdown_p90": float(np.percentile([r.max_drawdown for r in runs], 90)),
    }

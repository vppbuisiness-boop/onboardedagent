"""Pick the best slip at every price the book offers, and rank the prices.

For each ladder (power and flex, every size the book sells) the engine builds the highest-EV slip it can from the
bettable board: independent legs (one per match) priced with the Poisson-binomial, and same-team stacks priced
with the copula's full hit-count distribution, so flex ladders get credit for partial hits. Every candidate gets
the growth-optimal (Kelly) stake for its ladder, computed on the whole payout distribution, and the ranking uses
the growth rate rather than raw EV: a 37x ladder with a 5% hit rate can have a higher EV and a lower growth rate
than a 10x flex that pays on half its slips.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..ev.math import poisson_binomial_pmf
from ..ev.payouts import available_slips, ladder
from ..models.copula import Component, joint_hit_pmf
from ..models.predict import load_models
from .builder import build
from .stacks import _lines


@dataclass
class Candidate:
    book: str
    slip_type: str
    size: int
    kind: str  # 'independent' | 'same-team stack'
    legs: list[dict]
    pmf: np.ndarray
    net: list[float]
    link: str | None = None
    label: str = ""
    ev: float = field(init=False)
    p_top: float = field(init=False)
    p_paid: float = field(init=False)
    kelly: float = field(init=False)
    growth: float = field(init=False)

    stake_pmf: np.ndarray | None = None  # conservative distribution used for the stake (legs shrunk, stack lift halved)

    def __post_init__(self):
        net = np.asarray(self.net, dtype=float)
        self.ev = float(np.dot(self.pmf, net))
        self.p_top = float(self.pmf[-1])
        self.p_paid = float(self.pmf[net > 0].sum())
        self.kelly, self.growth = kelly_fraction(self.stake_pmf if self.stake_pmf is not None else self.pmf, net)


def kelly_fraction(pmf: np.ndarray, net: np.ndarray, hi: float = 0.999) -> tuple[float, float]:
    """Stake fraction that maximises E[log(1 + f * net)] over the payout distribution, and the growth rate at it.
    net is the net return per dollar for each hit count (-1 = lose the stake). Returns (0, 0) when EV <= 0."""
    pmf = np.asarray(pmf, dtype=float); net = np.asarray(net, dtype=float)
    if float(np.dot(pmf, net)) <= 0:
        return 0.0, 0.0

    def g(f: float) -> float:
        return float(np.sum(pmf * np.log1p(f * net)))

    lo, up = 0.0, hi
    for _ in range(80):
        m1, m2 = lo + (up - lo) / 3, up - (up - lo) / 3
        if g(m1) < g(m2):
            lo = m1
        else:
            up = m2
    f = (lo + up) / 2
    return float(f), g(f)


def _leg_dict(r, line: float) -> dict:
    return {"projection_id": r.projection_id, "player": r.player_name, "team": r.team, "opponent": r.opponent, "sport": r.sport,
            "stat_type": r.stat_type, "line": float(line), "lean": r.lean, "prob": float(r.prob), "start_time": r.start_time}


def best_slips(conn: sqlite3.Connection, book: str = "prizepicks", sports: list[str] | None = None, min_leg: float = 0.55,
               n_sim: int = 4000, max_per_game: int = 1, shrink: float = 0.05, days: int | None = None) -> list[Candidate]:
    """`shrink` is taken off every leg's probability for the stake only (the model's 60%+ calls have run at their
    label on real lines, but two days of calibration do not justify staking on the point estimate); for stacks the
    correlation credit is halved for the stake, since the lift is a backtest against fair lines."""
    from ..books.prizepicks import tail_link

    out: list[Candidate] = []
    # independent legs: the builder's best slip per ladder
    for slip_type, size in available_slips(book):
        slips = build(conn, book, slip_type, size, max_slips=1, sports=sports, max_per_game=max_per_game, days=days)
        for sl in slips:
            probs = [l["prob"] for l in sl.legs]
            net = ladder(book, slip_type, size)
            out.append(Candidate(book, slip_type, size, "independent", sl.legs, poisson_binomial_pmf(probs), net, sl.link,
                                 label=" + ".join(f"{l['player']} {l['lean'][0].lower()}{l['line']:g}" for l in sl.legs),
                                 stake_pmf=poisson_binomial_pmf([max(0.0, q - shrink) for q in probs])))
    # same-team stacks: one stat model, copula hit-count distribution, every ladder at that size
    df = _lines(conn, book, sports, min_leg, days)
    if not df.empty:
        df = df[~df["notes"].fillna("").str.contains("market_unproven")]
        models_by_sport: dict[str, dict] = {}
        sizes = sorted({n for _, n in available_slips(book)})
        for (sport, stat, gid, side, team), g in df.groupby(["sport", "stat", "game_id", "lean", "team"]):
            if len(g) < min(sizes):
                continue
            models = models_by_sport.setdefault(sport, load_models(sport))
            model = models.get(stat)
            if model is None:
                continue
            cand = g.sort_values("prob", ascending=False)
            for k in sizes:
                legs = cand.head(k)
                if len(legs) < k:
                    continue
                comps = []
                for r in legs.itertuples():
                    maps = list(range(int(r.map_from or 1), int(r.map_to or r.map_from or 1) + 1))
                    mu = float(r.projection) / len(maps) if r.projection is not None and not pd.isna(r.projection) else float(r.current_line) / len(maps)
                    comps.append(([Component(mu=mu, player=r.player_name, team=r.team, map_index=m) for m in maps], float(r.current_line), side))
                pmf = joint_hit_pmf(comps, model.r, model.rho_self, model.rho_team, model.rho_opp, n=n_sim)
                leg_dicts = [_leg_dict(r, r.current_line) for r in legs.itertuples()]
                link = tail_link([(r.projection_id, "o" if side == "OVER" else "u", float(r.current_line)) for r in legs.itertuples()]) if book == "prizepicks" else None
                label = f"{team} {stat} {side.lower()}s: " + " + ".join(f"{l['player']} {l['line']:g}" for l in leg_dicts)
                indep_shrunk = poisson_binomial_pmf([max(0.0, float(q) - shrink) for q in legs["prob"]])
                stake_pmf = 0.5 * pmf + 0.5 * indep_shrunk
                for slip_type, n in available_slips(book):
                    if n == k:
                        out.append(Candidate(book, slip_type, k, "same-team stack", leg_dicts, pmf, ladder(book, slip_type, k), link, label,
                                             stake_pmf=stake_pmf))
    out.sort(key=lambda c: -c.growth)
    return out


def best_per_price(cands: list[Candidate]) -> list[Candidate]:
    seen: dict[tuple[str, int], Candidate] = {}
    for c in cands:  # already sorted by growth
        seen.setdefault((c.slip_type, c.size), c)
    return sorted(seen.values(), key=lambda c: -c.growth)


def format_table(cands: list[Candidate], bankroll: float, kelly_fraction_used: float, cap: float) -> str:
    rows = []
    for c in cands:
        stake = min(c.kelly * kelly_fraction_used, cap) * bankroll if c.growth > 0 else 0.0
        rows.append({"price": f"{c.slip_type} {c.size} ({ladder(c.book, c.slip_type, c.size)[-1] + 1:g}x)", "kind": c.kind, "EV/$": f"{c.ev:+.0%}",
                     "P(top)": f"{c.p_top:.1%}", "P(paid)": f"{c.p_paid:.0%}", "growth/slip": f"{c.growth:+.2%}", "full Kelly": f"{c.kelly:.1%}",
                     f"stake @{kelly_fraction_used:g} Kelly": f"${stake:,.0f}", "slip": c.label[:90]})
    return pd.DataFrame(rows).to_string(index=False)

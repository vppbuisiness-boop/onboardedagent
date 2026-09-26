"""Same-match stacks: legs from one game, one stat and one direction, priced jointly with the copula.

A fixed parlay payout does not price correlation. Kills on one map move together (rounds played, pace), so a
stack of unders (or overs) from one match has a joint probability well above the product of its legs; the
walk-forward stack backtest on CS2 kills found realized joint hit rates about double the independent product,
with the copula's joint estimate within two points of realized. Correlation multiplies the legs' edge, so
stacks belong in markets whose legs are proven on real lines (Dota, LoL first).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..ev.payouts import ladder
from ..models.copula import Component, joint_hit_probability
from ..models.predict import load_models
from .builder import _candidates  # noqa: F401  (kept for parity; stacks use their own query so lower-probability legs qualify)


@dataclass
class Stack:
    book: str
    sport: str
    stat: str
    game_id: str
    side: str
    legs: list[dict]
    joint: float
    independent: float
    ev: float
    link: str | None

    @property
    def size(self) -> int:
        return len(self.legs)


def _lines(conn: sqlite3.Connection, book: str, sports: list[str] | None, min_leg: float) -> pd.DataFrame:
    q = """SELECT p.projection_id, p.lean, p.prob, p.ev, p.projection, p.notes, l.sport, l.stat, l.player_name, l.team, l.opponent, l.game_id,
                  l.stat_type, l.start_time, l.current_line, l.map_from, l.map_to
           FROM predictions p JOIN lines l ON l.book=p.book AND l.projection_id=p.projection_id
           WHERE p.book=? AND p.lean IS NOT NULL AND p.prob >= ? AND l.combo=0 AND l.game_id IS NOT NULL
             AND l.start_time > strftime('%Y-%m-%dT%H:%M:%SZ','now') AND (l.current_odds_type IS NULL OR l.current_odds_type='standard')
             AND (p.notes IS NULL OR (p.notes NOT LIKE '%voidable%' AND p.notes NOT LIKE '%banned%' AND p.notes NOT LIKE '%bumped_against%'))"""
    params: list = [book, min_leg]
    if sports:
        q += f" AND l.sport IN ({','.join('?' * len(sports))})"; params += sports
    return pd.read_sql_query(q, conn, params=params)


def stack_slips(conn: sqlite3.Connection, book: str = "prizepicks", sports: list[str] | None = None, sizes=(3, 4, 5), min_leg: float = 0.55,
                max_stacks: int = 10, n_sim: int = 4000) -> list[Stack]:
    from ..books.prizepicks import tail_link

    df = _lines(conn, book, sports, min_leg)
    if df.empty:
        return []
    models_by_sport: dict[str, dict] = {}
    out: list[Stack] = []
    for (sport, stat, gid, side), g in df.groupby(["sport", "stat", "game_id", "lean"]):
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
            joint, _ = joint_hit_probability(comps, model.r, model.rho_self, model.rho_team, model.rho_opp, n=n_sim)
            indep = float(np.prod(legs["prob"].to_numpy(dtype=float)))
            payout = ladder(book, "POWER", k)[-1] + 1.0
            leg_dicts = [{"projection_id": r.projection_id, "player": r.player_name, "team": r.team, "opponent": r.opponent, "stat_type": r.stat_type,
                          "line": float(r.current_line), "lean": r.lean, "prob": float(r.prob), "projection": None if pd.isna(r.projection) else float(r.projection),
                          "start_time": r.start_time} for r in legs.itertuples()]
            link = tail_link([(r.projection_id, "o" if side == "OVER" else "u", float(r.current_line)) for r in legs.itertuples()]) if book == "prizepicks" else None
            out.append(Stack(book, sport, stat, gid, side, leg_dicts, float(joint), indep, float(joint * payout - 1.0), link))
    out.sort(key=lambda s: -s.ev)
    return out[:max_stacks]


def format_stack(s: Stack, idx: int) -> str:
    head = f"[{idx}] {s.sport.upper()} {s.stat} {s.side} stack, {s.size} legs, one match: joint {s.joint:.1%} (independent {s.independent:.1%}), EV {s.ev:+.0%} at {s.size}-pick power"
    lines = [head]
    for l in s.legs:
        lines.append(f"     {l['player']:<16s} {l['stat_type']:<22s} {l['lean']:<5s} {l['line']:<5g} p={l['prob']:.1%} proj={'' if l['projection'] is None else round(l['projection'], 1)}  {l['team']} vs {l['opponent']}  {l['start_time'][:16]}")
    if s.link:
        lines.append(f"     {s.link}")
    return "\n".join(lines)

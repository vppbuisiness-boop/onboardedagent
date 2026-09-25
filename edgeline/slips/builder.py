"""Greedy EV-optimal slip builder with exposure constraints.

Rules (defaults mirror LCSLarry's product): only bettable legs, at most one leg per
player and one per game per slip, skip lines already marked used, rank by leg EV,
compute slip EV with the exact Poisson-binomial over the book's payout ladder.
Pushes are treated as losses inside a slip (conservative; a push actually shrinks
the slip on PrizePicks).
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import dataclass

import pandas as pd

from ..books.prizepicks import tail_link
from ..ev.math import poisson_binomial_pmf, slip_ev_from_pmf, slip_hit_prob
from ..ev.payouts import ladder


@dataclass
class Slip:
    book: str
    slip_type: str
    legs: list[dict]
    ev: float
    hit_prob: float
    link: str | None

    @property
    def size(self) -> int:
        return len(self.legs)


def _candidates(conn: sqlite3.Connection, book: str, sports: list[str] | None) -> pd.DataFrame:
    q = """
    SELECT p.*, l.sport, l.player_name, l.team, l.opponent, l.game_id, l.stat_type, l.start_time, l.current_line, l.open_line,
           l.current_odds_type, l.voidable
    FROM predictions p JOIN lines l ON l.book=p.book AND l.projection_id=p.projection_id
    WHERE p.book=? AND p.bettable=1
      AND NOT EXISTS (SELECT 1 FROM used_lines u WHERE u.book=p.book AND u.projection_id=p.projection_id)
    """
    df = pd.read_sql_query(q, conn, params=(book,))
    if sports:
        df = df[df["sport"].isin(sports)]
    # keep the latest model version per projection
    df = df.sort_values("computed_at").groupby("projection_id", as_index=False).tail(1)
    return df.sort_values(["ev", "prob"], ascending=False).reset_index(drop=True)


def build(conn: sqlite3.Connection, book: str = "prizepicks", slip_type: str = "POWER", size: int = 3, max_slips: int = 10,
          sports: list[str] | None = None, max_per_game: int = 1, max_per_player: int = 1, min_slip_ev: float = 0.0,
          rank_by: str = "ev") -> list[Slip]:
    cands = _candidates(conn, book, sports)
    if rank_by == "prob":
        cands = cands.sort_values(["prob", "ev"], ascending=False).reset_index(drop=True)
    net = ladder(book, slip_type, size)
    used_ids: set[str] = set()
    slips: list[Slip] = []
    while len(slips) < max_slips:
        legs, games, players = [], {}, {}
        for r in cands.itertuples(index=False):
            if r.projection_id in used_ids:
                continue
            if games.get(r.game_id, 0) >= max_per_game or players.get(r.player_name, 0) >= max_per_player:
                continue
            legs.append(r)
            games[r.game_id] = games.get(r.game_id, 0) + 1
            players[r.player_name] = players.get(r.player_name, 0) + 1
            if len(legs) == size:
                break
        if len(legs) < size:
            break
        probs = [float(l.prob) for l in legs]
        pmf = poisson_binomial_pmf(probs)
        ev = slip_ev_from_pmf(pmf, net)
        if ev < min_slip_ev:
            break
        hp = slip_hit_prob(probs, book, slip_type)
        leg_dicts = [
            {
                "projection_id": l.projection_id, "player": l.player_name, "team": l.team, "opponent": l.opponent, "sport": l.sport,
                "stat_type": l.stat_type, "line": float(l.current_line), "lean": l.lean, "prob": float(l.prob), "ev": float(l.ev),
                "projection": None if l.projection is None else float(l.projection), "start_time": l.start_time,
            }
            for l in legs
        ]
        link = tail_link([(l.projection_id, "o" if l.lean == "OVER" else "u", float(l.current_line)) for l in legs]) if book == "prizepicks" else None
        slips.append(Slip(book, slip_type, leg_dicts, float(ev), float(hp), link))
        used_ids.update(l.projection_id for l in legs)
    return slips


def save_slips(conn: sqlite3.Connection, slips: list[Slip]) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    conn.executemany(
        "INSERT INTO slips(created_at, book, slip_type, size, ev, hit_prob, legs_json, link) VALUES (?,?,?,?,?,?,?,?)",
        [(now, s.book, s.slip_type, s.size, s.ev, s.hit_prob, json.dumps(s.legs), s.link) for s in slips],
    )
    conn.commit()


def mark_used(conn: sqlite3.Connection, book: str, projection_ids: list[str]) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    conn.executemany("INSERT OR IGNORE INTO used_lines(book, projection_id, used_at) VALUES (?,?,?)", [(book, p, now) for p in projection_ids])
    conn.commit()


def format_slip(s: Slip, idx: int) -> str:
    head = f"[{idx}] {s.book} {s.slip_type} {s.size}-pick  EV {s.ev:+.1%}  hit {s.hit_prob:.1%}  payout {ladder(s.book, s.slip_type, s.size)[-1] + 1:g}x"
    body = "\n".join(
        f"     {l['sport'].upper():5s} {l['player']:<18s} {l['stat_type']:<22s} {l['lean']:<5s} {l['line']:<5g} p={l['prob']:.1%} ev={l['ev']:+.1%} proj={l['projection'] if l['projection'] is None else round(l['projection'],1)}  {l['team']} vs {l['opponent']}"
        for l in s.legs
    )
    return head + "\n" + body + (f"\n     {s.link}" if s.link else "")

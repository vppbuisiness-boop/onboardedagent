"""Line shopping: the same player and stat across books, with the book whose line and payout give the model the most EV."""
from __future__ import annotations

import re
import sqlite3

import pandas as pd


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def shop(conn: sqlite3.Connection, sport: str | None = None, upcoming_only: bool = True) -> pd.DataFrame:
    """One row per (sport, player, stat type, day) posted by more than one book.

    Columns per book: line, lean, probability, EV; plus `best_book` (highest EV for that book's own lean) and the
    line spread. Books price the same lean differently when their lines differ, so the model's edge is largest on the
    book with the friendlier number."""
    q = """SELECT l.book, l.sport, l.player_name, l.team, l.stat_type, l.current_line, l.odds_over, l.odds_under, l.start_time,
                  p.lean, p.prob, p.ev, p.bettable
           FROM lines l JOIN predictions p ON p.book=l.book AND p.projection_id=l.projection_id
           WHERE p.lean IS NOT NULL"""
    params: list = []
    if sport:
        q += " AND l.sport=?"; params.append(sport)
    if upcoming_only:
        q += " AND l.start_time > strftime('%Y-%m-%dT%H:%M:%SZ','now')"
    df = pd.read_sql_query(q, conn, params=params)
    if df.empty:
        return df
    df["day"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    df["pkey"] = df["player_name"].map(_key)
    rows = []
    for (sp, pk, st, day), g in df.groupby(["sport", "pkey", "stat_type", "day"]):
        if g["book"].nunique() < 2:
            continue
        rec = {"sport": sp, "player": g["player_name"].iloc[0], "team": g["team"].iloc[0], "stat_type": st, "day": day}
        best, best_ev = None, -9
        for r in g.itertuples(index=False):
            rec[f"{r.book} line"] = r.current_line
            rec[f"{r.book} lean"] = r.lean
            rec[f"{r.book} prob"] = round(float(r.prob), 3)
            rec[f"{r.book} EV"] = round(float(r.ev), 3)
            if r.ev > best_ev:
                best, best_ev = r.book, float(r.ev)
        rec["best_book"] = best
        rec["best_EV"] = round(best_ev, 3)
        rec["line_spread"] = round(float(g["current_line"].max() - g["current_line"].min()), 1)
        rec["bettable_anywhere"] = int(g["bettable"].max())
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.sort_values(["bettable_anywhere", "best_EV"], ascending=[False, False]) if len(out) else out

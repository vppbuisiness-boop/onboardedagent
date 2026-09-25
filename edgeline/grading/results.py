"""Public-tracker style results: hit rate, ROI at implied leg odds, and 4-pick parlay ROI.

Conventions follow LCSLarry's results page so numbers are comparable:
- a prediction counts once per unique line (latest model version);
- 'open' grades the model's lean against the opening line, 'current' against the last seen line;
- leg P&L uses the book's 4-pick POWER implied per-leg odds; pushes return the stake;
- parlay ROI = (hit_rate ** 4) * payout - 1, i.e. betting every 4 legs as one 4-pick power.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from ..ev.payouts import ladder, leg_decimal_odds


def results_frame(conn: sqlite3.Connection, book: str = "prizepicks", min_prob: float = 0.0, bettable_only: bool = False) -> pd.DataFrame:
    q = """
    SELECT p.book, p.projection_id, p.model_version, p.lean, p.prob, p.ev, p.bettable, p.computed_at,
           l.sport, l.player_name, l.stat_type, l.start_time, l.open_line, l.current_line,
           g.actual, g.result_open, g.result_current
    FROM predictions p
    JOIN lines l ON l.book=p.book AND l.projection_id=p.projection_id
    JOIN grades g ON g.book=p.book AND g.projection_id=p.projection_id
    WHERE p.book=? AND p.lean IS NOT NULL
    """
    df = pd.read_sql_query(q, conn, params=(book,))
    if df.empty:
        return df
    df = df.sort_values("computed_at").groupby("projection_id", as_index=False).tail(1)
    df = df[df["prob"] >= min_prob]
    if bettable_only:
        df = df[df["bettable"] == 1]
    odds = leg_decimal_odds(book)
    for col, res in (("open", "result_open"), ("current", "result_current")):
        lean = df["lean"].str.lower()
        df[f"win_{col}"] = (df[res] == lean).astype(float)
        df.loc[df[res].isin(["push", "void"]), f"win_{col}"] = float("nan")
        df[f"pnl_{col}"] = df[f"win_{col}"].map({1.0: odds - 1.0, 0.0: -1.0}).fillna(0.0)
    return df


def summarize(df: pd.DataFrame, book: str = "prizepicks", by: str | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    payout = ladder(book, "POWER", 4)[-1] + 1.0
    keys = [by] if by else []

    def agg(g: pd.DataFrame) -> pd.Series:
        out = {"n": len(g)}
        for col in ("open", "current"):
            settled = g[f"win_{col}"].dropna()
            hr = settled.mean() if len(settled) else float("nan")
            out[f"wins_{col}"] = int(settled.sum()) if len(settled) else 0
            out[f"losses_{col}"] = int(len(settled) - settled.sum()) if len(settled) else 0
            out[f"hit_{col}"] = hr
            out[f"leg_roi_{col}"] = g[f"pnl_{col}"].sum() / max(len(settled), 1)
            out[f"parlay4_roi_{col}"] = (hr**4) * payout - 1.0 if len(settled) else float("nan")
        return pd.Series(out)

    if keys:
        return df.groupby(keys).apply(agg, include_groups=False).reset_index()
    return agg(df).to_frame().T

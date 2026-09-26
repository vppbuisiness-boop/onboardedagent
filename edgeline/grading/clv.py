"""Closing-line value (CLV): does the book's own line move toward our side after we price a pick?

Settled results need hundreds of picks before a 60% leg rate separates from the 56.2% break-even.
CLV is the faster signal sportsbook bettors use: if the market repeatedly moves toward the side we
took (an UNDER whose line later drops, an OVER whose line later rises), the book's own closing
judgement agrees with the model. It is not profit, but a consistent positive CLV is the strongest
early evidence of an edge, and a negative one is a warning no backtest can override.

Method
- one row per line, graded the way the results tracker grades 'open': the lean is the model's
  projection (latest pre-game pricing) against the opening line we captured, so a line that moved
  past our projection before the close still counts as the side we would have taken at the open;
- closing line = the last snapshot recorded before the scheduled start (lines leave the board when
  the game starts, so this is the book's last word);
- only started games count (the line can no longer move);
- 'for' = the close moved toward our lean, 'against' = away; unchanged lines are reported but
  excluded from the sign test, whose p-value is the exact two-sided binomial test at 50%;
- 'edge' bins the projected gap at the open (|projection - open line|), so the report shows whether
  the book moves more often toward us when the model disagrees with it more.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
from scipy.stats import binomtest


def clv_frame(conn: sqlite3.Connection, book: str = "prizepicks", now: pd.Timestamp | None = None) -> pd.DataFrame:
    q = """
    SELECT p.book, p.projection_id, p.model_version, p.computed_at, p.line AS pick_line, p.projection, p.lean, p.prob, p.ev, p.bettable,
           l.sport, l.player_name, l.stat_type, l.start_time, l.open_line, l.current_line
    FROM predictions p
    JOIN lines l ON l.book=p.book AND l.projection_id=p.projection_id
    WHERE p.book=? AND p.lean IS NOT NULL AND l.start_time IS NOT NULL
    """
    df = pd.read_sql_query(q, conn, params=(book,))
    if df.empty:
        return df
    df = df.sort_values("computed_at").groupby("projection_id", as_index=False).tail(1)
    df["start"] = pd.to_datetime(df["start_time"], utc=True, format="ISO8601")
    df["computed"] = pd.to_datetime(df["computed_at"], utc=True, format="ISO8601")
    now = now or pd.Timestamp.now(tz="UTC")
    df = df[df["start"] <= now].copy()
    if df.empty:
        return df
    snaps = pd.read_sql_query(
        "SELECT projection_id, fetched_at, line FROM line_snapshots WHERE book=? ORDER BY fetched_at", conn, params=(book,)
    )
    snaps["fetched"] = pd.to_datetime(snaps["fetched_at"], utc=True, format="ISO8601")
    starts = df.set_index("projection_id")["start"]
    snaps = snaps[snaps["projection_id"].isin(starts.index)]
    snaps = snaps[snaps["fetched"] <= snaps["projection_id"].map(starts)]
    close = snaps.groupby("projection_id")["line"].last()
    df["close_line"] = df["projection_id"].map(close).fillna(df["current_line"])
    proj = df["projection"]
    lean_open = pd.Series("UNDER", index=df.index).where(proj < df["open_line"], "OVER")
    df["lean_open"] = lean_open.where(proj.notna(), df["lean"].str.upper())
    sign = df["lean_open"].map({"UNDER": -1.0, "OVER": 1.0})
    df["move"] = df["close_line"] - df["open_line"]
    df["clv"] = df["move"] * sign  # >0: the close moved toward our side
    df["clv_pct"] = df["clv"] / df["open_line"].where(df["open_line"] != 0)
    gap = (proj - df["open_line"]).abs()
    df["edge"] = pd.cut(gap, [-0.01, 0.5, 1.0, 2.0, 1e9], labels=["<=0.5", "0.5-1", "1-2", ">2"])
    df["flipped"] = df["lean_open"] != df["lean"].str.upper()
    return df


def _agg(g: pd.DataFrame) -> pd.Series:
    n = len(g)
    moved = g[g["clv"] != 0]
    n_for = int((moved["clv"] > 0).sum())
    n_against = int((moved["clv"] < 0).sum())
    k = n_for + n_against
    share = n_for / k if k else float("nan")
    pval = binomtest(n_for, k, 0.5).pvalue if k else float("nan")
    return pd.Series({
        "n": n, "unchanged": n - k, "for": n_for, "against": n_against,
        "for_share": share, "p_value": pval,
        "mean_clv": g["clv"].mean(), "mean_clv_pct": g["clv_pct"].mean(),
    })


def summarize(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if by:
        return df.groupby(by).apply(_agg, include_groups=False).reset_index()
    return _agg(df).to_frame().T


def report(df: pd.DataFrame, by: str | None = None) -> str:
    if df.empty:
        return "no started lines with a pick yet"
    parts = []
    for label, sub in (("all model leans", df), ("bettable picks (>=60% prob, >=5% EV)", df[df["bettable"] == 1])):
        if sub.empty:
            continue
        parts.append(f"== {label}")
        table = summarize(sub, by).copy()
        table["for_share"] = (table["for_share"] * 100).round(1)
        table["mean_clv"] = table["mean_clv"].round(3)
        table["mean_clv_pct"] = (table["mean_clv_pct"] * 100).round(2)
        table["p_value"] = table["p_value"].round(3)
        parts.append(table.to_string(index=False))
    parts.append(f"lean flipped between open and close on {int(df['flipped'].sum())} of {len(df)} lines")
    parts.append("for/against: the close moved toward/away from the side our projection took at the open; for_share in %,")
    parts.append("p_value = exact binomial vs 50%; mean_clv in line units (+ = toward us), mean_clv_pct = as % of the opening line.")
    return "\n".join(parts)

"""Data-driven player ban list.

A player is banned when their graded model picks underperform the model's own
stated probabilities by a statistically significant margin (one-sided binomial
test against the mean predicted probability of the picks). Mirrors the rule the
original product describes: "banned only if there's statistically significant
evidence they're underperforming expectation".
"""
from __future__ import annotations

import datetime as dt
import sqlite3

import pandas as pd
from scipy import stats


def evaluate(df: pd.DataFrame, min_n: int = 15, alpha: float = 0.05) -> pd.DataFrame:
    """df columns: sport, player_name, prob (model prob of the lean), hit (1/0). Returns per-player table."""
    if df.empty:
        return pd.DataFrame(columns=["sport", "player_name", "n", "hits", "expected", "pvalue", "banned"])
    rows = []
    for (sport, player), g in df.groupby(["sport", "player_name"]):
        n = int(len(g))
        hits = int(g["hit"].sum())
        p = float(g["prob"].mean())
        pvalue = float(stats.binom.cdf(hits, n, p))  # P(X <= hits) under the model's own claim
        rows.append({"sport": sport, "player_name": player, "n": n, "hits": hits, "expected": p * n, "pvalue": pvalue,
                     "banned": int(n >= min_n and pvalue < alpha)})
    return pd.DataFrame(rows).sort_values(["banned", "pvalue"], ascending=[False, True]).reset_index(drop=True)


def graded_picks(conn: sqlite3.Connection, book: str = "prizepicks") -> pd.DataFrame:
    q = """
    SELECT l.sport, l.player_name, p.prob, p.lean, g.result_open AS result
    FROM predictions p
    JOIN lines l ON l.book=p.book AND l.projection_id=p.projection_id
    JOIN grades g ON g.book=p.book AND g.projection_id=p.projection_id
    WHERE p.book=? AND p.lean IS NOT NULL AND p.prob IS NOT NULL AND g.result_open IN ('over','under')
    """
    df = pd.read_sql_query(q, conn, params=(book,))
    if df.empty:
        return df
    df["hit"] = (df["result"] == df["lean"].str.lower()).astype(int)
    return df[["sport", "player_name", "prob", "hit"]]


def update(conn: sqlite3.Connection, book: str = "prizepicks", min_n: int = 15, alpha: float = 0.05) -> pd.DataFrame:
    table = evaluate(graded_picks(conn, book), min_n, alpha)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    conn.execute("DELETE FROM banned_players")
    banned = table[table["banned"] == 1] if not table.empty else table
    conn.executemany(
        "INSERT INTO banned_players(sport, player_name, n, hits, expected, pvalue, banned_at) VALUES (?,?,?,?,?,?,?)",
        [(r.sport, r.player_name, int(r.n), int(r.hits), float(r.expected), float(r.pvalue), now) for r in banned.itertuples(index=False)],
    )
    conn.commit()
    return table


def banned_set(conn: sqlite3.Connection, sport: str) -> set[str]:
    return {r[0] for r in conn.execute("SELECT player_name FROM banned_players WHERE sport=?", (sport,)).fetchall()}

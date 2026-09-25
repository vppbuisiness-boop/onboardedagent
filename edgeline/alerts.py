"""Discord webhook alerts for newly bettable lines (the 'line drop alert' loop).

Set EDGELINE_DISCORD_WEBHOOK to a Discord webhook URL. Each bettable projection is
alerted once; sent ids are recorded in alerts_sent.
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3

import pandas as pd
import requests

from .books.prizepicks import tail_link

WEBHOOK_ENV = "EDGELINE_DISCORD_WEBHOOK"


def pending_alerts(conn: sqlite3.Connection, book: str = "prizepicks") -> pd.DataFrame:
    q = """
    SELECT p.projection_id, p.lean, p.prob, p.ev, p.projection, p.line, p.computed_at,
           l.sport, l.player_name, l.team, l.opponent, l.stat_type, l.start_time, l.open_line
    FROM predictions p JOIN lines l ON l.book=p.book AND l.projection_id=p.projection_id
    WHERE p.book=? AND p.bettable=1
      AND NOT EXISTS (SELECT 1 FROM alerts_sent a WHERE a.book=p.book AND a.projection_id=p.projection_id)
    ORDER BY p.ev DESC
    """
    df = pd.read_sql_query(q, conn, params=(book,))
    if df.empty:
        return df
    return df.sort_values("computed_at").groupby("projection_id", as_index=False).tail(1).sort_values("ev", ascending=False)


def format_message(df: pd.DataFrame, book: str = "prizepicks") -> str:
    lines = [f"**{len(df)} new bettable line{'s' if len(df) != 1 else ''} on {book}**"]
    for r in df.itertuples(index=False):
        link = tail_link([(r.projection_id, "o" if r.lean == "OVER" else "u", float(r.line))]) if book == "prizepicks" else ""
        proj = f" proj {r.projection:.1f}" if r.projection is not None and not pd.isna(r.projection) else ""
        lines.append(
            f"`{r.sport.upper():4s}` {r.player_name} ({r.team} vs {r.opponent}) {r.stat_type} **{r.lean} {r.line:g}**"
            f" p={r.prob:.0%} EV={r.ev:+.0%}{proj} {link}"
        )
    return "\n".join(lines)[:1900]


def send(conn: sqlite3.Connection, book: str = "prizepicks", webhook: str | None = None, dry_run: bool = False) -> int:
    df = pending_alerts(conn, book)
    if df.empty:
        return 0
    webhook = webhook or os.environ.get(WEBHOOK_ENV)
    msg = format_message(df, book)
    if dry_run:
        print(msg)
        return int(len(df))
    if not webhook:
        print(msg)
        print(f"(no webhook configured; set {WEBHOOK_ENV}; nothing marked as sent)")
        return 0
    r = requests.post(webhook, json={"content": msg}, timeout=30)
    r.raise_for_status()
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    conn.executemany("INSERT OR IGNORE INTO alerts_sent(book, projection_id, sent_at) VALUES (?,?,?)",
                     [(book, pid, now) for pid in df["projection_id"]])
    conn.commit()
    return int(len(df))

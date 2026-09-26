"""Export and import the tables that cannot be re-downloaded: captured lines, snapshots, predictions, grades, slips.

History tables are re-loadable from their sources; the line record is not. Keep data/exports/ in git.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import DATA_DIR

EXPORT_DIR = DATA_DIR / "exports"
TABLES = ["lines", "line_snapshots", "predictions", "grades", "slips", "used_lines", "banned_players", "alerts_sent", "kalshi_quotes", "kalshi_grades"]


def _ensure_tables(conn: sqlite3.Connection) -> None:
    from .books.kalshi import SCHEMA  # the Kalshi tables are created by the scanner, not by db.connect

    conn.executescript(SCHEMA)


def export_tables(conn: sqlite3.Connection, out_dir: Path = EXPORT_DIR) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_tables(conn)
    counts = {}
    for t in TABLES:
        df = pd.read_sql_query(f"SELECT * FROM {t}", conn)
        df.to_csv(out_dir / f"{t}.csv", index=False)
        counts[t] = int(len(df))
    return counts


def import_tables(conn: sqlite3.Connection, in_dir: Path = EXPORT_DIR) -> dict[str, int]:
    _ensure_tables(conn)
    counts = {}
    for t in TABLES:
        p = in_dir / f"{t}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if df.empty:
            counts[t] = 0
            continue
        if "id" in df.columns and t in ("line_snapshots", "slips"):
            df = df.drop(columns=["id"])
        cols = list(df.columns)
        rows = [tuple(None if pd.isna(v) else v for v in r) for r in df.itertuples(index=False, name=None)]
        conn.executemany(f"INSERT OR IGNORE INTO {t}({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})", rows)
        counts[t] = len(rows)
    conn.commit()
    return counts

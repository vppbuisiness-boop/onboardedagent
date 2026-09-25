"""SQLite storage. Append-only line snapshots plus derived tables.

Tables
- line_snapshots: every observation of every line (book, projection id, line, odds type, time).
- lines: one row per (book, projection_id) with the OPENING line (first seen) and the CURRENT line.
- player_games: one row per player per map/game from history sources (sport-agnostic).
- predictions: model output per line and model version.
- grades: settled outcome per line.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS line_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book TEXT NOT NULL,
    projection_id TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    line REAL NOT NULL,
    odds_type TEXT,
    status TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_snap_proj ON line_snapshots(book, projection_id, fetched_at);

CREATE TABLE IF NOT EXISTS lines (
    book TEXT NOT NULL,
    projection_id TEXT NOT NULL,
    sport TEXT NOT NULL,
    league TEXT,
    player_id TEXT,
    player_name TEXT NOT NULL,
    team TEXT,
    opponent TEXT,
    position TEXT,
    game_id TEXT,
    stat_type TEXT NOT NULL,
    stat TEXT,
    map_from INTEGER,
    map_to INTEGER,
    combo INTEGER DEFAULT 0,
    combo_players TEXT,
    voidable INTEGER DEFAULT 0,
    board_time TEXT,
    start_time TEXT,
    open_line REAL NOT NULL,
    open_seen_at TEXT NOT NULL,
    current_line REAL NOT NULL,
    current_odds_type TEXT,
    last_seen_at TEXT NOT NULL,
    status TEXT,
    PRIMARY KEY (book, projection_id)
);
CREATE INDEX IF NOT EXISTS ix_lines_sport_start ON lines(sport, start_time);

CREATE TABLE IF NOT EXISTS player_games (
    sport TEXT NOT NULL,
    source TEXT NOT NULL,
    game_id TEXT NOT NULL,
    series_id TEXT,
    game_number INTEGER,
    date TEXT NOT NULL,
    league TEXT,
    tier TEXT,
    patch TEXT,
    player_name TEXT NOT NULL,
    player_id TEXT,
    team TEXT,
    opponent TEXT,
    role TEXT,
    side TEXT,
    champion TEXT,
    kills REAL,
    deaths REAL,
    assists REAL,
    headshots REAL,
    team_kills REAL,
    opp_kills REAL,
    game_length REAL,
    win INTEGER,
    playoffs INTEGER,
    PRIMARY KEY (sport, source, game_id, player_name)
);
CREATE INDEX IF NOT EXISTS ix_pg_player ON player_games(sport, player_name, date);
CREATE INDEX IF NOT EXISTS ix_pg_team ON player_games(sport, team, date);

CREATE TABLE IF NOT EXISTS predictions (
    book TEXT NOT NULL,
    projection_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    computed_at TEXT NOT NULL,
    line REAL NOT NULL,
    projection REAL,
    p_over REAL,
    p_under REAL,
    ev_over REAL,
    ev_under REAL,
    lean TEXT,
    prob REAL,
    ev REAL,
    bettable INTEGER DEFAULT 0,
    notes TEXT,
    PRIMARY KEY (book, projection_id, model_version)
);

CREATE TABLE IF NOT EXISTS grades (
    book TEXT NOT NULL,
    projection_id TEXT NOT NULL,
    graded_at TEXT NOT NULL,
    actual REAL,
    maps_played INTEGER,
    result_open TEXT,
    result_current TEXT,
    source TEXT,
    PRIMARY KEY (book, projection_id)
);

CREATE TABLE IF NOT EXISTS slips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    book TEXT NOT NULL,
    slip_type TEXT NOT NULL,
    size INTEGER NOT NULL,
    ev REAL,
    hit_prob REAL,
    legs_json TEXT NOT NULL,
    link TEXT
);

CREATE TABLE IF NOT EXISTS used_lines (
    book TEXT NOT NULL,
    projection_id TEXT NOT NULL,
    used_at TEXT NOT NULL,
    PRIMARY KEY (book, projection_id)
);
"""


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def session(path: Path | str = DB_PATH):
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

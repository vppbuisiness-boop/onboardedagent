"""Settle book lines against history rows once matches are over.

For a line (player, stat, maps a..b, start_time), find that player's games within a
window around the start time; pick the series closest to the start time; sum the stat
over game_number in [a, b]. If fewer maps were played than the scope needs and the prop
is voidable, grade it 'void'. Results are computed against both the opening and the
current (closing) line.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3

import pandas as pd

from ..features.build import current_state
from ..models.predict import NameResolver, resolve_board_teams


def _outcome(actual: float, line: float) -> str:
    if actual > line:
        return "over"
    if actual < line:
        return "under"
    return "push"


def grade_lines(conn: sqlite3.Connection, sport: str, book: str = "prizepicks", min_age_hours: float = 4.0, window_hours: float = 30.0) -> dict:
    now = pd.Timestamp.now(tz="UTC")
    lines = pd.read_sql_query(
        "SELECT * FROM lines WHERE book=? AND sport=? AND projection_id NOT IN (SELECT projection_id FROM grades WHERE book=?)",
        conn, params=(book, sport, book),
    )
    if lines.empty:
        return {"graded": 0, "void": 0, "pending": 0}
    lines["start_dt"] = pd.to_datetime(lines["start_time"], utc=True, errors="coerce")
    due = lines[lines["start_dt"] < now - pd.Timedelta(hours=min_age_hours)]
    pg = pd.read_sql_query("SELECT * FROM player_games WHERE sport=?", conn, params=(sport,))
    if pg.empty or due.empty:
        return {"graded": 0, "void": 0, "pending": int(len(lines))}
    pg["date"] = pd.to_datetime(pg["date"], utc=True, errors="coerce")
    resolver = NameResolver(pg["player_name"].unique().tolist(), pg["team"].dropna().unique().tolist())
    player_state, _ = current_state(pg)
    team_map = resolve_board_teams(due, player_state, resolver)
    graded = void = 0
    graded_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for ln in due.itertuples(index=False):
        names = json.loads(ln.combo_players) if ln.combo and ln.combo_players else [ln.player_name]
        resolved = [resolver.player(n) for n in names]
        if any(r is None for r in resolved) or ln.stat not in pg.columns:
            continue
        opp = team_map.get(ln.opponent)
        totals, maps_played_min = 0.0, None
        ok = True
        for pname in resolved:
            g = pg[(pg["player_name"] == pname) & (pg["date"] >= ln.start_dt - pd.Timedelta(hours=6)) & (pg["date"] <= ln.start_dt + pd.Timedelta(hours=window_hours))]
            if opp is not None and (g["opponent"] == opp).any():
                g = g[g["opponent"] == opp]
            if g.empty:
                ok = False
                break
            # choose the series whose first game is closest to start_time
            first = g.groupby("series_id")["date"].min()
            sid = (first - ln.start_dt).abs().idxmin()
            series = g[g["series_id"] == sid].sort_values("game_number")
            played = int(series["game_number"].max())
            maps_played_min = played if maps_played_min is None else min(maps_played_min, played)
            sel = series[(series["game_number"] >= ln.map_from) & (series["game_number"] <= ln.map_to)]
            totals += float(pd.to_numeric(sel[ln.stat], errors="coerce").fillna(0).sum())
        if not ok:
            continue
        if maps_played_min is not None and maps_played_min < ln.map_to:
            result_open = result_cur = "void"
            void += 1
        else:
            result_open = _outcome(totals, float(ln.open_line))
            result_cur = _outcome(totals, float(ln.current_line))
        conn.execute(
            "INSERT OR REPLACE INTO grades(book, projection_id, graded_at, actual, maps_played, result_open, result_current, source) VALUES (?,?,?,?,?,?,?,?)",
            (book, ln.projection_id, graded_at, totals, maps_played_min, result_open, result_cur, "player_games"),
        )
        graded += 1
    conn.commit()
    return {"graded": graded, "void": void, "pending": int(len(lines) - graded)}

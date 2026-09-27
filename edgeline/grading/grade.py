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
from ..features.build import canonical_teams
from ..models.predict import NameResolver, resolve_board_teams, rosters

# A map counts toward a grade only when the source has published it in full. bo3.gg serves some maps with a partial
# stats table (a 20-round map with 55 kills across ten players) and lists one-round technical restarts as map 1 of a
# series, so a line graded on those maps is graded on a false zero. Every map in a line's range must have a full roster
# and a plausible kill total, and technical maps are dropped before maps are numbered.
MIN_PLAYERS = {"cs2": 10, "val": 10, "lol": 10, "dota": 10, "cod": 8}
MIN_GAME_KILLS = {"cs2": 60, "val": 60, "lol": 8, "dota": 12, "cod": 40}
JUNK_MAX_ROUNDS = 3  # a CS2 or Valorant "map" of three rounds or fewer is a restart, not a map
REGRADE_DAYS = 3.0  # grades this young are recomputed every pass, so late corrections to the history flow through


def game_completeness(pg: pd.DataFrame, sport: str) -> pd.DataFrame:
    """One row per game_id: roster size, total kills, rounds, whether the map is a technical restart, and whether it is
    complete enough to grade."""
    g = pg.groupby("game_id").agg(players=("player_name", "nunique"), total_kills=("kills", "sum"), rounds=("rounds", "max"))
    rounds = pd.to_numeric(g["rounds"], errors="coerce")
    g["junk"] = (rounds.notna() & (rounds <= JUNK_MAX_ROUNDS)) if sport in ("cs2", "val") else False
    g["complete"] = (~g["junk"]) & (g["players"] >= MIN_PLAYERS.get(sport, 10)) & (g["total_kills"] >= MIN_GAME_KILLS.get(sport, 0))
    return g


def drop_junk_maps(pg: pd.DataFrame, gc: pd.DataFrame) -> pd.DataFrame:
    """Remove technical maps and renumber the remaining maps of the series they came from, in start order."""
    junk = set(gc.index[gc["junk"]])
    if not junk:
        return pg
    touched = set(pg.loc[pg["game_id"].isin(junk), "series_id"])
    pg = pg[~pg["game_id"].isin(junk)].copy()
    sub = pg[pg["series_id"].isin(touched)]
    order = sub.groupby(["series_id", "game_id"])["date"].min().reset_index().sort_values(["series_id", "date"])
    order["new_number"] = order.groupby("series_id").cumcount() + 1
    renum = dict(zip(zip(order["series_id"], order["game_id"]), order["new_number"]))
    idx = pg["series_id"].isin(touched)
    pg.loc[idx, "game_number"] = [renum[(s, g)] for s, g in zip(pg.loc[idx, "series_id"], pg.loc[idx, "game_id"])]
    return pg


def _outcome(actual: float, line: float) -> str:
    if actual > line:
        return "over"
    if actual < line:
        return "under"
    return "push"


def series_over(series: pd.DataFrame, now: pd.Timestamp, settle_hours: float) -> bool:
    """True when the loaded maps show a decided series: three map wins for one side, or two map wins once
    `settle_hours` have passed since the last loaded map (a 2-0 could still be a best-of-five in progress).
    Sources publish maps one at a time, so a partially loaded series must not void later-map lines."""
    per_map = series.groupby("game_number")["win"].max()
    wins_me = int((per_map == 1).sum())
    wins_opp = int((per_map == 0).sum())
    lead = max(wins_me, wins_opp)
    if lead >= 3:
        return True
    last = series["date"].max()
    return lead >= 2 and pd.notna(last) and (last + pd.Timedelta(hours=settle_hours)) < now


def grade_lines(conn: sqlite3.Connection, sport: str, book: str = "prizepicks", min_age_hours: float = 4.0, window_hours: float = 30.0,
                settle_hours: float = 6.0, regrade_days: float | None = REGRADE_DAYS) -> dict:
    now = pd.Timestamp.now(tz="UTC")
    if regrade_days:
        # recompute young grades from the current history: sources correct partial maps for a day or two after a match
        since = (now - pd.Timedelta(days=regrade_days)).strftime("%Y-%m-%dT%H:%M:%S")
        conn.execute(
            "DELETE FROM grades WHERE book=? AND projection_id IN (SELECT projection_id FROM lines WHERE book=? AND sport=? AND substr(start_time,1,19) >= ?)",
            (book, book, sport, since),
        )
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
    pg = canonical_teams(pg)
    completeness = game_completeness(pg, sport)
    pg = drop_junk_maps(pg, completeness)
    complete_games = set(completeness.index[completeness["complete"]])
    player_state, _ = current_state(pg)
    resolver = NameResolver(pg["player_name"].unique().tolist(), pg["team"].dropna().unique().tolist(), rosters(player_state))
    team_map = resolve_board_teams(due, player_state, resolver)
    graded = void = 0
    graded_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for ln in due.itertuples(index=False):
        names = json.loads(ln.combo_players) if ln.combo and ln.combo_players else [ln.player_name]
        resolved = [resolver.player(n, team_map.get(ln.team)) for n in names]
        if any(r is None for r in resolved) or ln.stat not in pg.columns:
            continue
        opp = team_map.get(ln.opponent)
        totals, maps_played_min = 0.0, None
        ok, decided = True, True
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
            if played < ln.map_to and not series_over(series, now, settle_hours):
                decided = False  # later maps may still be coming; leave the line pending
            sel = series[(series["game_number"] >= ln.map_from) & (series["game_number"] <= ln.map_to)]
            vals = pd.to_numeric(sel[ln.stat], errors="coerce")
            if sel["game_number"].nunique() < min(played, ln.map_to) - ln.map_from + 1 or not set(sel["game_id"]) <= complete_games or vals.isna().any():
                decided = False  # a map in the range is missing, partial or has no value for this stat: wait for the source, never grade a false zero
            totals += float(vals.sum())
        if not ok or not decided:
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

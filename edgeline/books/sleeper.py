"""Sleeper Picks board client (public JSON, no headers needed).

Endpoints: /lines/available?dynamic=true (every active pick with per-side payout multipliers), /players/{sport}
(subject id -> in-game name and team), /schedule/{sport}/regular/{year} (game id -> teams and date). Sleeper
posts esports props for Counter-Strike as `kills_maps_1_2` and `headshots_maps_1_2` today; other esports are
mapped when they appear. Each option carries its own multiplier (1.78 standard), stored per side as odds so the
pricer computes EV against the actual payout of that pick.
"""
from __future__ import annotations

import datetime as dt
import re
import sqlite3
import time

import requests

from ..config import USER_AGENT
from .prizepicks import LineRecord, upsert_lines

API = "https://api.sleeper.app"
BOOK = "sleeper"
SPORTS = {"cs": "cs2", "lol": "lol", "valorant": "val", "val": "val", "dota2": "dota", "dota": "dota", "cod": "cod"}
STAT_RE = re.compile(r"^(kills|headshots|deaths|assists)_maps?_(\d+)(?:_(\d+))?$")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s


def fetch(session: requests.Session | None = None) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    """(lines, players by sleeper sport, schedule by sleeper sport) for the esports Sleeper lists."""
    s = session or _session()
    lines = s.get(f"{API}/lines/available", params={"dynamic": "true", "include_preseason": "true"}, timeout=60).json()
    lines = [ln for ln in lines if ln.get("sport") in SPORTS]
    sports = sorted({ln["sport"] for ln in lines})
    players, schedule = {}, {}
    year = dt.date.today().year
    for sp in sports:
        try:
            players[sp] = s.get(f"{API}/players/{sp}", timeout=60).json()
        except Exception:
            players[sp] = {}
        try:
            schedule[sp] = {g["game_id"]: g for g in s.get(f"{API}/schedule/{sp}/regular/{year}", timeout=60).json()}
        except Exception:
            schedule[sp] = {}
        time.sleep(0.5)
    return lines, players, schedule


def _stat_type(stat: str, a: int, b: int) -> str:
    label = stat.capitalize()
    return f"MAP {a} {label}" if a == b else f"MAPS {a}-{b} {label}"


def parse(lines: list[dict], players: dict[str, dict], schedule: dict[str, dict], start_times: dict[tuple, str] | None = None) -> list[tuple[LineRecord, float | None, float | None]]:
    """(record, odds_over, odds_under) per pick. Start time is the match date at noon UTC unless `start_times`
    (keyed by (sport, frozenset(teams), date)) supplies a better one, e.g. from PrizePicks' board."""
    out = []
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for ln in lines:
        sp = SPORTS.get(ln.get("sport"))
        m = STAT_RE.match(ln.get("wager_type") or "")
        if not sp or not m:
            continue
        stat, a, b = m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))
        opts = {o.get("outcome"): o for o in ln.get("options", []) if o.get("status") == "active"}
        if "over" not in opts and "under" not in opts:
            continue
        o = opts.get("over") or opts.get("under")
        line = o.get("outcome_value")
        if line is None:
            continue
        pl = (players.get(ln["sport"]) or {}).get(str(ln.get("subject_id"))) or {}
        name = pl.get("username") or (pl.get("metadata") or {}).get("username") or f"{pl.get('first_name', '')} {pl.get('last_name', '')}".strip() or f"sleeper:{ln.get('subject_id')}"
        team = o.get("subject_team") or pl.get("team")
        g = (schedule.get(ln["sport"]) or {}).get(ln.get("game_id")) or {}
        home, away = (g.get("home") or {}).get("name"), (g.get("away") or {}).get("name")
        opponent = None
        if team and home and away:
            opponent = away if team == home else home if team == away else None
        date = g.get("date")
        start = None
        if date:
            key = (sp, frozenset(x for x in (home, away) if x), date)
            start = (start_times or {}).get(key) or f"{date}T12:00:00Z"
        status = {"pre_game": "pre_game", "in_progress": "in_game", "complete": "final"}.get(o.get("game_status"), o.get("game_status"))
        rec = LineRecord(book=BOOK, projection_id=f"{ln.get('game_id')}:{ln.get('market_type')}", sport=sp, league=None, player_id=str(ln.get("subject_id")),
                         player_name=name, team=team, opponent=opponent, position=None, game_id=ln.get("game_id"), stat_type=_stat_type(stat, a, b), stat=stat,
                         map_from=a, map_to=b, combo=0, combo_players=None, voidable=int(b > a), board_time=None, start_time=start, line=float(line),
                         odds_type="standard", status=status, updated_at=now)
        mo = float(opts["over"]["payout_multiplier"]) if "over" in opts and opts["over"].get("payout_multiplier") else None
        mu = float(opts["under"]["payout_multiplier"]) if "under" in opts and opts["under"].get("payout_multiplier") else None
        out.append((rec, mo, mu))
    return out


def prizepicks_start_times(conn: sqlite3.Connection) -> dict[tuple, str]:
    """Match start times PrizePicks already posted, keyed by (sport, teams, UTC date), to time Sleeper's date-only games."""
    rows = conn.execute("SELECT sport, team, opponent, start_time FROM lines WHERE book='prizepicks' AND start_time IS NOT NULL AND opponent IS NOT NULL").fetchall()
    out: dict[tuple, str] = {}
    for sport, team, opp, st in rows:
        try:
            date = dt.datetime.fromisoformat(st.replace("Z", "+00:00")).astimezone(dt.timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            continue
        out[(sport, frozenset([team, opp]), date)] = st
    return out


def pull(conn: sqlite3.Connection) -> dict:
    lines, players, schedule = fetch()
    parsed = parse(lines, players, schedule, prizepicks_start_times(conn))
    by_sport: dict[str, list] = {}
    for rec, mo, mu in parsed:
        by_sport.setdefault(rec.sport, []).append((rec, mo, mu))
    summary = {}
    for sport, items in by_sport.items():
        res = upsert_lines(conn, [r for r, _, _ in items])
        conn.executemany("UPDATE lines SET odds_over=?, odds_under=? WHERE book=? AND projection_id=?", [(mo, mu, BOOK, r.projection_id) for r, mo, mu in items])
        conn.commit()
        summary[sport] = {"projections": len(items), **res}
    return summary

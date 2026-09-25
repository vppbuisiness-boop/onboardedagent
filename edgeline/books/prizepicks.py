"""PrizePicks board client (partner API host, JSON:API format).

The consumer host api.prizepicks.com sits behind DataDome; partner-api.prizepicks.com
serves the same projections without it. Fields we rely on:
  attributes.line_score, stat_type, odds_type ('standard' | 'demon' | 'goblin'),
  board_time (when the line was posted), start_time, status, updated_at
  relationships.new_player -> included new_player {name, team, position, combo}
  relationships.game       -> included game {external_game_id, start_time}
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass

import requests

from ..config import USER_AGENT
from .normalize import opponent_from_game_id, parse_stat_type, voidable

PARTNER_API = "https://partner-api.prizepicks.com"
LEAGUE_IDS = {"lol": 121, "cs2": 265, "val": 159, "dota": 174, "cod": 145, "r6": 274}
BOOK = "prizepicks"


@dataclass
class LineRecord:
    book: str
    projection_id: str
    sport: str
    league: str | None
    player_id: str | None
    player_name: str
    team: str | None
    opponent: str | None
    position: str | None
    game_id: str | None
    stat_type: str
    stat: str | None
    map_from: int | None
    map_to: int | None
    combo: int
    combo_players: str | None
    voidable: int
    board_time: str | None
    start_time: str | None
    line: float
    odds_type: str | None
    status: str | None
    updated_at: str | None


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s


def _get_with_retry(s: requests.Session, url: str, params: dict, tries: int = 5, base_sleep: float = 2.0) -> requests.Response:
    """GET with exponential backoff on 429/5xx (the partner host throttles bursts)."""
    last: requests.Response | None = None
    for attempt in range(tries):
        r = s.get(url, params=params, timeout=60)
        if r.status_code < 400:
            return r
        last = r
        if r.status_code not in (429, 500, 502, 503, 504):
            r.raise_for_status()
        retry_after = r.headers.get("Retry-After")
        wait = float(retry_after) if retry_after and retry_after.isdigit() else base_sleep * (2 ** attempt)
        time.sleep(min(wait, 60.0))
    assert last is not None
    last.raise_for_status()
    return last


def fetch_board(sport: str, per_page: int = 250, session: requests.Session | None = None, pause: float = 6.0) -> dict:
    """Fetch all pages for a sport; returns {'data': [...], 'included': [...]}."""
    league_id = LEAGUE_IDS[sport]
    s = session or _session()
    data, included, page, total = [], [], 1, 1
    while page <= total:
        r = _get_with_retry(
            s,
            f"{PARTNER_API}/projections",
            {
                "league_id": league_id,
                "per_page": per_page,
                "page": page,
                "single_stat": "true",
                "include": "new_player,league,stat_type,game",
            },
        )
        payload = r.json()
        data.extend(payload.get("data", []))
        included.extend(payload.get("included", []))
        total = int((payload.get("meta") or {}).get("total_pages", 1) or 1)
        page += 1
        if page <= total:
            time.sleep(pause)
    return {"data": data, "included": included}


def _split_combo_names(name: str) -> list[str]:
    parts = re.split(r"\s*\+\s*", name or "")
    return [p.strip() for p in parts if p.strip()]


def parse_board(payload: dict, sport: str) -> list[LineRecord]:
    inc = {(i["type"], str(i["id"])): i for i in payload.get("included", [])}
    records: list[LineRecord] = []
    for p in payload.get("data", []):
        a = p.get("attributes", {})
        rel = p.get("relationships", {})
        pl_ref = (rel.get("new_player") or {}).get("data") or {}
        gm_ref = (rel.get("game") or {}).get("data") or {}
        player = inc.get(("new_player", str(pl_ref.get("id")))) or {}
        game = inc.get(("game", str(gm_ref.get("id")))) or {}
        pa = player.get("attributes", {})
        ga = game.get("attributes", {})
        scope = parse_stat_type(a.get("stat_type") or "")
        name = pa.get("name") or pa.get("display_name") or ""
        combo = bool(pa.get("combo")) or (scope.combo if scope else False) or ("+" in name)
        line = a.get("line_score")
        if line is None or not name:
            continue
        rec = LineRecord(
            book=BOOK,
            projection_id=str(p["id"]),
            sport=sport,
            league=a.get("league_ppid"),
            player_id=str(pl_ref.get("id")) if pl_ref else None,
            player_name=name,
            team=pa.get("team") or None,
            opponent=None,
            position=pa.get("position") or None,
            game_id=ga.get("external_game_id") or a.get("game_id"),
            stat_type=a.get("stat_type") or "",
            stat=scope.stat if scope else None,
            map_from=scope.map_from if scope else None,
            map_to=scope.map_to if scope else None,
            combo=int(combo),
            combo_players=json.dumps(_split_combo_names(name)) if combo else None,
            voidable=int(voidable(sport, scope)) if scope else 0,
            board_time=a.get("board_time"),
            start_time=a.get("start_time") or ga.get("start_time"),
            line=float(line),
            odds_type=a.get("odds_type"),
            status=a.get("status"),
            updated_at=a.get("updated_at"),
        )
        records.append(rec)

    # Resolve opponents: two team codes appear per game across the board.
    teams_by_game: dict[str, set[str]] = {}
    for r in records:
        if r.game_id and r.team:
            teams_by_game.setdefault(r.game_id, set()).add(r.team)
    for r in records:
        teams = teams_by_game.get(r.game_id or "", set())
        others = [t for t in teams if t != r.team]
        if len(others) == 1:
            r.opponent = others[0]
        else:
            r.opponent = opponent_from_game_id(r.game_id or "", r.team or "")
    return records


def upsert_lines(conn: sqlite3.Connection, records: list[LineRecord], fetched_at: str | None = None) -> dict:
    """Append snapshots and maintain open/current line per projection. Returns counts."""
    fetched_at = fetched_at or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    new, updated, moved = 0, 0, 0
    for r in records:
        conn.execute(
            "INSERT INTO line_snapshots(book, projection_id, fetched_at, line, odds_type, status, updated_at) VALUES (?,?,?,?,?,?,?)",
            (r.book, r.projection_id, fetched_at, r.line, r.odds_type, r.status, r.updated_at),
        )
        row = conn.execute(
            "SELECT current_line FROM lines WHERE book=? AND projection_id=?", (r.book, r.projection_id)
        ).fetchone()
        if row is None:
            d = asdict(r)
            conn.execute(
                """INSERT INTO lines(book, projection_id, sport, league, player_id, player_name, team, opponent, position,
                       game_id, stat_type, stat, map_from, map_to, combo, combo_players, voidable, board_time, start_time,
                       open_line, open_seen_at, current_line, current_odds_type, last_seen_at, status)
                   VALUES (:book,:projection_id,:sport,:league,:player_id,:player_name,:team,:opponent,:position,
                       :game_id,:stat_type,:stat,:map_from,:map_to,:combo,:combo_players,:voidable,:board_time,:start_time,
                       :line,:fetched_at,:line,:odds_type,:fetched_at,:status)""",
                {**d, "fetched_at": fetched_at},
            )
            new += 1
        else:
            if abs(float(row["current_line"]) - r.line) > 1e-9:
                moved += 1
            conn.execute(
                """UPDATE lines SET current_line=?, current_odds_type=?, last_seen_at=?, status=?, start_time=COALESCE(?, start_time),
                       opponent=COALESCE(opponent, ?)
                   WHERE book=? AND projection_id=?""",
                (r.line, r.odds_type, fetched_at, r.status, r.start_time, r.opponent, r.book, r.projection_id),
            )
            updated += 1
    return {"new": new, "updated": updated, "moved": moved, "fetched_at": fetched_at}


def pull(conn: sqlite3.Connection, sports: list[str], pause: float = 12.0) -> dict:
    """Pull and persist each sport's board; commits after every sport so a throttled request can't lose earlier work."""
    s = _session()
    summary = {}
    for i, sport in enumerate(sports):
        if i:
            time.sleep(pause)
        try:
            payload = fetch_board(sport, session=s)
        except requests.HTTPError as exc:
            summary[sport] = {"projections": 0, "new": 0, "updated": 0, "moved": 0, "error": str(exc)}
            continue
        recs = parse_board(payload, sport)
        summary[sport] = {"projections": len(recs), **upsert_lines(conn, recs)}
        conn.commit()
    return summary


def tail_link(legs: list[tuple[str, str, float]]) -> str:
    """Build a PrizePicks deep link. legs: (projection_id, 'o'|'u', line)."""
    parts = [f"{pid}-{side}-{line:g}" for pid, side, line in legs]
    return "https://app.prizepicks.com/?projections=" + ",".join(parts)

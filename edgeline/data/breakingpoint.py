"""Call of Duty (and CS2) history from Breaking Point's public Supabase REST backend.

breakingpoint.gg ships a Supabase anon key in its web bundle and its stats tables allow
anonymous reads, so per-map player stats are one paged query away:
  player_stats (game_id, match_id, player_tag, team_id, kills, deaths, assists, mode_id, map_id, datetime)
  games        (id, match_id, game_num, mode_id, map_id, team_1_id, team_2_id, team_1_kills, team_2_kills, winner_id, gametime_*)
  matches      (id, datetime, best_of, team_1_id, team_2_id, winner_id, event_id)
  teams, maps, modes (id -> names; modes carry sport_id: 1 = Call of Duty, 2 = CS2)
Override the key with EDGELINE_BREAKINGPOINT_KEY if it rotates.
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
import time

import pandas as pd
import requests

from ..config import USER_AGENT
from .opendota import write_player_games

HOST = "https://dfpiiufxcciujugzjvgx.supabase.co/rest/v1"
DEFAULT_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRmcGlpdWZ"
    "4Y2NpdWp1Z3pqdmd4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ2ODk0MDMsImV4cCI6MjA2MDI2NTQ"
    "wM30.36VuOTvrxtmR3nb-u3nnVYWzMBn9YP1bQFvUYF5T1OE"
)
SPORT_IDS = {"cod": 1, "cs2": 2}
PAGE = 1000


def _session() -> requests.Session:
    key = os.environ.get("EDGELINE_BREAKINGPOINT_KEY", DEFAULT_KEY)
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "apikey": key, "Authorization": f"Bearer {key}", "Accept": "application/json"})
    return s


def _select(s: requests.Session, table: str, params: dict, pause: float = 0.2) -> list[dict]:
    """Paged select via Range headers."""
    out, start = [], 0
    while True:
        r = s.get(f"{HOST}/{table}", params=params, headers={"Range": f"{start}-{start + PAGE - 1}", "Prefer": "count=none"}, timeout=60)
        if r.status_code not in (200, 206):
            raise RuntimeError(f"breakingpoint {r.status_code} on {table}: {r.text[:200]}")
        rows = r.json()
        out.extend(rows)
        if len(rows) < PAGE:
            break
        start += PAGE
        time.sleep(pause)
    return out


def fetch(sport: str, since: dt.date, until: dt.date | None = None) -> pd.DataFrame:
    s = _session()
    sport_id = SPORT_IDS[sport]
    modes = {m["id"]: m for m in _select(s, "modes", {"select": "id,name,short_name,sport_id", "sport_id": f"eq.{sport_id}"})}
    if not modes:
        return pd.DataFrame()
    mode_ids = ",".join(str(i) for i in modes)
    params = {"select": "game_id,match_id,player_id,player_tag,team_id,kills,deaths,assists,mode_id,map_id,datetime,event_id",
              "mode_id": f"in.({mode_ids})", "datetime": f"gte.{since.isoformat()}T00:00:00Z", "order": "datetime.asc,game_id.asc", "deleted_at": "is.null"}
    if until:
        params["and"] = f"(datetime.lte.{until.isoformat()}T23:59:59Z)"
    stats = pd.DataFrame(_select(s, "player_stats", params))
    if stats.empty:
        return stats
    match_ids = sorted(set(stats["match_id"].dropna().astype(int)))
    games, matches = [], []
    for i in range(0, len(match_ids), 200):
        chunk = ",".join(map(str, match_ids[i:i + 200]))
        games.extend(_select(s, "games", {"select": "id,match_id,game_num,mode_id,map_id,team_1_id,team_2_id,team_1_kills,team_2_kills,team_1_score,team_2_score,winner_id,gametime_min,gametime_sec",
                                          "match_id": f"in.({chunk})", "deleted_at": "is.null"}))
        matches.extend(_select(s, "matches", {"select": "id,datetime,best_of,team_1_id,team_2_id,winner_id,event_id", "id": f"in.({chunk})"}))
    games = pd.DataFrame(games).drop_duplicates("id").set_index("id")
    matches = pd.DataFrame(matches).drop_duplicates("id").set_index("id")
    team_ids = sorted(set(stats["team_id"].dropna().astype(int)) | set(games["team_1_id"].dropna().astype(int)) | set(games["team_2_id"].dropna().astype(int)))
    teams = {}
    for i in range(0, len(team_ids), 300):
        chunk = ",".join(map(str, team_ids[i:i + 300]))
        for t in _select(s, "teams", {"select": "id,name,name_short", "id": f"in.({chunk})"}):
            teams[t["id"]] = t
    maps = {m["id"]: m for m in _select(s, "maps", {"select": "id,name"})}
    rows = []
    for st in stats.itertuples(index=False):
        g = games.loc[st.game_id] if st.game_id in games.index else None
        if g is None:
            continue
        m = matches.loc[st.match_id] if st.match_id in matches.index else None
        team = teams.get(int(st.team_id), {}).get("name") if pd.notna(st.team_id) else None
        t1, t2 = g["team_1_id"], g["team_2_id"]
        opp_id = t2 if st.team_id == t1 else t1
        opponent = teams.get(int(opp_id), {}).get("name") if pd.notna(opp_id) else None
        tk, ok = (g["team_1_kills"], g["team_2_kills"]) if st.team_id == t1 else (g["team_2_kills"], g["team_1_kills"])
        length = None
        if pd.notna(g.get("gametime_min")):
            length = float(g["gametime_min"]) + (float(g["gametime_sec"]) if pd.notna(g.get("gametime_sec")) else 0) / 60.0
        mode = modes.get(int(st.mode_id), {}) if pd.notna(st.mode_id) else {}
        rows.append({
            "sport": sport, "source": "breakingpoint", "game_id": str(st.game_id), "series_id": str(int(st.match_id)), "game_number": int(g["game_num"] or 1),
            "date": st.datetime, "league": f"event{int(st.event_id)}" if pd.notna(st.event_id) else None, "tier": None, "patch": None,
            "player_name": st.player_tag, "player_id": str(int(st.player_id)) if pd.notna(st.player_id) else None, "team": team, "opponent": opponent,
            "role": mode.get("short_name") or "unknown", "side": None, "champion": maps.get(int(st.map_id), {}).get("name") if pd.notna(st.map_id) else None,
            "kills": st.kills, "deaths": st.deaths, "assists": st.assists, "headshots": None, "team_kills": tk, "opp_kills": ok,
            "game_length": length, "win": int(g["winner_id"] == st.team_id) if pd.notna(g.get("winner_id")) else None,
            "playoffs": None,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df


def load(conn: sqlite3.Connection, sport: str, since: dt.date, until: dt.date | None = None) -> dict:
    df = fetch(sport, since, until)
    if df.empty:
        return {"rows": 0, "written": 0}
    n = write_player_games(conn, df)
    conn.commit()
    return {"rows": int(len(df)), "written": n, "maps": int(df["game_id"].nunique()), "matches": int(df["series_id"].nunique())}

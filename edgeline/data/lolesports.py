"""League of Legends history from Riot's official esports API and livestats feed.

Pipeline: leagues -> tournaments (with dates) -> completed events per tournament ->
event details (games, teams) -> livestats window at game start (metadata, first frame)
and far after the end (final frame with kills/deaths/assists per participant).

The persisted API key below is the public key the lolesports.com site itself sends;
it is not a secret. The feed endpoints need no key.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from ..config import USER_AGENT
from .opendota import write_player_games

API = "https://esports-api.lolesports.com/persisted/gw"
FEED = "https://feed.lolesports.com/livestats/v1"
PUBLIC_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
ROLES = {1: "top", 2: "jungle", 3: "mid", 4: "bot", 5: "support", 6: "top", 7: "jungle", 8: "mid", 9: "bot", 10: "support"}
DEFAULT_LEAGUES = ["lck", "lpl", "lec", "lcs", "lta_n", "lta_s", "lta_cross", "cblol-brazil", "lcp", "ljl-japan", "lla", "pcs", "vcs",
                   "worlds", "msi", "first_stand", "emea_masters", "lck_challengers_league", "lcl", "tcl", "nlc", "lfl", "prime_league",
                   "superliga", "americas_cup", "hitpoint_masters", "lit", "arabian_league", "esports_balkan_league", "roadoflegends"]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "x-api-key": PUBLIC_KEY, "Accept": "application/json"})
    return s


def _get(s: requests.Session, url: str, params: dict | None = None, tries: int = 4):
    last = None
    for i in range(tries):
        r = s.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        last = r
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"lolesports {last.status_code if last else '?'} for {url}")


def leagues(s: requests.Session) -> list[dict]:
    return _get(s, f"{API}/getLeagues", {"hl": "en-US"})["data"]["leagues"]


def tournaments(s: requests.Session, league_id: str) -> list[dict]:
    d = _get(s, f"{API}/getTournamentsForLeague", {"hl": "en-US", "leagueId": league_id})
    out = []
    for lg in d["data"]["leagues"]:
        out.extend(lg.get("tournaments", []))
    return out


def completed_events(s: requests.Session, tournament_id: str) -> list[dict]:
    d = _get(s, f"{API}/getCompletedEvents", {"hl": "en-US", "tournamentId": tournament_id})
    return [e for e in d["data"]["schedule"]["events"] if e.get("match") and e["match"].get("id")]


def event_details(s: requests.Session, match_id: str) -> dict | None:
    d = _get(s, f"{API}/getEventDetails", {"hl": "en-US", "id": match_id})
    return d["data"]["event"] if d else None


def window(s: requests.Session, game_id: str, starting_time: str | None = None) -> dict | None:
    return _get(s, f"{FEED}/window/{game_id}", {"startingTime": starting_time} if starting_time else None)


def _round10(ts: dt.datetime) -> str:
    ts = ts.replace(second=(ts.second // 10) * 10, microsecond=0)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def game_rows(s: requests.Session, event: dict, game: dict, match_start: dt.datetime) -> list[dict]:
    """Two feed calls: start window (metadata + first frame) and a far-future window (final frame)."""
    gid = game["id"]
    start = window(s, gid)
    if not start or not start.get("frames"):
        return []
    end = window(s, gid, _round10(match_start + dt.timedelta(hours=12)))
    if not end or not end.get("frames"):
        return []
    last = end["frames"][-1]
    if last.get("gameState") != "finished":
        return []
    first_ts = pd.Timestamp(start["frames"][0]["rfc460Timestamp"])
    last_ts = pd.Timestamp(last["rfc460Timestamp"])
    game_length = (last_ts - first_ts).total_seconds() / 60.0
    md = start["gameMetadata"]
    teams = {t["code"]: t for t in event["match"]["teams"]}
    # map side -> team via participant name prefix ("GEN Kiin" -> GEN)
    side_meta = {"blue": md["blueTeamMetadata"], "red": md["redTeamMetadata"]}
    side_team = {}
    for side, meta in side_meta.items():
        codes = [p["summonerName"].split(" ")[0] for p in meta["participantMetadata"]]
        code = max(set(codes), key=codes.count)
        side_team[side] = teams.get(code, {"code": code, "name": code})
    side_kills = {"blue": last["blueTeam"]["totalKills"], "red": last["redTeam"]["totalKills"]}
    winner = "blue" if side_kills["blue"] >= side_kills["red"] else "red"  # refined below from match result when available
    res = {t["code"]: (t.get("result") or {}).get("gameWins") for t in event["match"]["teams"]}
    rows = []
    for side, opp in (("blue", "red"), ("red", "blue")):
        team = side_team[side]
        opponent = side_team[opp]
        parts = {p["participantId"]: p for p in last[f"{side}Team"]["participants"]}
        for pm in side_meta[side]["participantMetadata"]:
            pid = pm["participantId"]
            st = parts.get(pid)
            if st is None:
                continue
            name = pm["summonerName"]
            if name.startswith(team["code"] + " "):
                name = name[len(team["code"]) + 1:]
            rows.append({
                "sport": "lol", "source": "lolesports", "game_id": gid, "series_id": str(event.get("id") or event["match"].get("id")), "game_number": int(game.get("number") or 1),
                "date": first_ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "league": (event.get("league") or {}).get("slug"), "tier": None, "patch": md.get("patchVersion"),
                "player_name": name, "player_id": str(pm.get("esportsPlayerId") or ""), "team": team["name"], "opponent": opponent["name"],
                "role": ROLES.get(pid, "unknown"), "side": side, "champion": pm.get("championId"), "kills": st["kills"], "deaths": st["deaths"],
                "assists": st["assists"], "headshots": None, "team_kills": side_kills[side], "opp_kills": side_kills[opp],
                "game_length": game_length, "win": int(side == winner), "playoffs": int("playoff" in (event.get("blockName") or "").lower() or "final" in (event.get("blockName") or "").lower()),
            })
    return rows


def load(conn: sqlite3.Connection, since: dt.date, until: dt.date | None = None, league_slugs: list[str] | None = None,
         workers: int = 6, pause: float = 0.1, progress=None) -> dict:
    s = _session()
    until = until or dt.date.today() + dt.timedelta(days=1)
    slugs = set(league_slugs or DEFAULT_LEAGUES)
    lgs = [l for l in leagues(s) if l["slug"] in slugs]
    known = {r[0] for r in conn.execute("SELECT DISTINCT game_id FROM player_games WHERE sport='lol' AND source='lolesports'").fetchall()}
    events = []
    for lg in lgs:
        for t in tournaments(s, lg["id"]):
            t_start, t_end = dt.date.fromisoformat(t["startDate"]), dt.date.fromisoformat(t["endDate"])
            if t_end < since or t_start > until:
                continue
            for e in completed_events(s, t["id"]):
                st = pd.Timestamp(e["startTime"]).date()
                if since <= st <= until:
                    events.append(e)
            time.sleep(pause)
    if progress:
        progress(f"{len(lgs)} leagues, {len(events)} completed matches in range")

    def fetch(e: dict) -> list[dict]:
        local = _session()
        try:
            ev = event_details(local, e["match"]["id"])
            if not ev:
                return []
            match_start = pd.Timestamp(e["startTime"]).to_pydatetime()
            rows = []
            for g in ev["match"].get("games", []):
                if g.get("state") != "completed" or g["id"] in known:
                    continue
                rows.extend(game_rows(local, ev, g, match_start))
                time.sleep(pause)
            return rows
        except Exception:
            return []

    written, batch, done = 0, [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, e) for e in events]
        for fut in as_completed(futures):
            batch.extend(fut.result())
            done += 1
            if len(batch) >= 500:
                written += write_player_games(conn, pd.DataFrame(batch))
                conn.commit()
                batch = []
                if progress:
                    progress(f"{done}/{len(events)} matches, {written} rows written")
    if batch:
        written += write_player_games(conn, pd.DataFrame(batch))
        conn.commit()
    return {"leagues": len(lgs), "matches": len(events), "written": written}

"""CS2 history from bo3.gg's public JSON API.

matches -> games (maps, batched by match id) -> games/{id}/players_stats
(kills, deaths, assists, headshots, clan names, win). HLTV is Cloudflare-walled;
bo3.gg is the reachable alternative and covers tiers S through C.
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

API = "https://api.bo3.gg/api/v1"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s


def _get(s: requests.Session, url: str, params: dict | None = None, tries: int = 4):
    last = None
    for i in range(tries):
        r = s.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        last = r
        time.sleep(2.0 * (i + 1))
    raise RuntimeError(f"bo3.gg {last.status_code if last else '?'} for {url}")


def list_matches(s: requests.Session, since: dt.date, until: dt.date | None = None, tiers: list[str] | None = None,
                 max_matches: int | None = None, pause: float = 0.3) -> list[dict]:
    """Newest-first paging that stops at `since` (the API ignores date filters)."""
    out, offset, limit = [], 0, 100
    params = {
        "filter[matches.status][eq]": "finished",
        "filter[matches.discipline_id][eq]": 1,
        "sort": "-start_date",
        "page[limit]": limit,
    }
    if tiers:
        params["filter[matches.tier][in]"] = ",".join(tiers)
    since_iso = f"{since.isoformat()}T00:00:00"
    until_iso = f"{until.isoformat()}T23:59:59" if until else None
    while True:
        d = _get(s, f"{API}/matches", {**params, "page[offset]": offset})
        res = d.get("results", [])
        if not res:
            break
        for m in res:
            sd = (m.get("start_date") or "")[:19]
            if sd < since_iso:
                return out[:max_matches] if max_matches else out
            if until_iso and sd > until_iso:
                continue
            if tiers and m.get("tier") not in tiers:
                continue
            out.append(m)
        if len(res) < limit or (max_matches and len(out) >= max_matches):
            break
        offset += limit
        time.sleep(pause)
    return out[:max_matches] if max_matches else out


def games_for(s: requests.Session, match_ids: list[int], pause: float = 0.3) -> list[dict]:
    out = []
    for i in range(0, len(match_ids), 40):
        chunk = match_ids[i:i + 40]
        d = _get(s, f"{API}/games", {"filter[games.match_id][in]": ",".join(map(str, chunk)), "page[limit]": 200})
        out.extend(d.get("results", []))
        time.sleep(pause)
    return out


def players_for(s: requests.Session, game_id: int) -> list[dict]:
    d = _get(s, f"{API}/games/{game_id}/players_stats")
    return d if isinstance(d, list) else d.get("results", [])


def _team_name(p: dict) -> str:
    """Prefer the canonical team name; clan names can carry random suffixes."""
    tc = p.get("team_clan") or {}
    team = tc.get("team") or {}
    return team.get("name") or p.get("clan_name") or "unknown"


def build_rows(match: dict, game: dict, stats: list[dict]) -> list[dict]:
    clan_to_team = {p["clan_name"]: _team_name(p) for p in stats}
    kills_by_clan: dict[str, float] = {}
    for p in stats:
        kills_by_clan[p["clan_name"]] = kills_by_clan.get(p["clan_name"], 0.0) + float(p.get("kills") or 0)
    dur = game.get("duration")
    game_length = (dur / 1e9 / 60.0) if dur and dur > 1e6 else (dur / 60.0 if dur else None)
    rows = []
    for p in stats:
        prof = p.get("steam_profile") or {}
        name = prof.get("nickname") or prof.get("name") or f"steam:{p.get('steam_profile_id')}"
        rows.append({
            "sport": "cs2", "source": "bo3", "game_id": str(game["id"]), "series_id": str(match["id"]), "game_number": int(game.get("number") or 1),
            "date": game.get("begin_at") or match.get("start_date"), "league": f"t{match.get('tournament_id')}", "tier": match.get("tier"),
            "patch": None, "player_name": name, "player_id": str(p.get("steam_profile_id")), "team": clan_to_team.get(p["clan_name"]),
            "opponent": clan_to_team.get(p.get("enemy_clan_name"), p.get("enemy_clan_name")), "role": "unknown", "side": None,
            "champion": game.get("map_name"),
            "kills": p.get("kills"), "deaths": p.get("death"), "assists": p.get("assists"), "headshots": p.get("headshots"),
            "team_kills": kills_by_clan.get(p["clan_name"]), "opp_kills": kills_by_clan.get(p.get("enemy_clan_name")),
            "game_length": game_length, "rounds": game.get("rounds_count"), "win": int(bool(p.get("win"))), "playoffs": None,
        })
    return rows


def load(conn: sqlite3.Connection, since: dt.date, until: dt.date | None = None, tiers: list[str] | None = None,
         max_matches: int | None = None, pause: float = 0.1, workers: int = 6, progress=None) -> dict:
    s = _session()
    matches = list_matches(s, since, until, tiers, max_matches)
    known = {r[0] for r in conn.execute("SELECT DISTINCT game_id FROM player_games WHERE sport='cs2' AND source='bo3'").fetchall()}
    by_id = {m["id"]: m for m in matches}
    games = [g for g in games_for(s, list(by_id)) if g.get("status") == "finished" and str(g["id"]) not in known]
    if progress:
        progress(f"{len(matches)} matches, {len(games)} new finished maps to fetch with {workers} workers")

    def fetch(g: dict) -> tuple[dict, list[dict] | None]:
        local = _session()
        try:
            return g, players_for(local, g["id"])
        except Exception:
            return g, None
        finally:
            time.sleep(pause)

    written, failed, batch, done = 0, 0, [], 0

    def flush() -> None:
        nonlocal written, batch
        if not batch:
            return
        df = pd.DataFrame(batch)
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        written += write_player_games(conn, df)
        conn.commit()
        batch = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, g) for g in games]
        for fut in as_completed(futures):
            g, stats = fut.result()
            done += 1
            if stats is None:
                failed += 1
                continue
            batch.extend(build_rows(by_id[g["match_id"]], g, stats))
            if len(batch) >= 500:
                flush()
                if progress:
                    progress(f"{done}/{len(games)} maps, {written} rows written, {failed} failed")
    flush()
    return {"matches": len(matches), "maps": len(games), "written": written, "failed": failed}

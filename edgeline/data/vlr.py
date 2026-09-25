"""Valorant history from vlr.gg match pages (HTML; polite pacing).

Results archive: https://www.vlr.gg/matches/results?page=N (50 matches per page).
Match page: per-map blocks `div.vm-stats-game[data-game-id]` with `div.ovw-row` player
rows carrying kills/deaths/assists per side. Headshot counts are not published
(only HS%), so `headshots` stays null for Valorant.
"""
from __future__ import annotations

import datetime as dt
import re
import sqlite3
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..config import USER_AGENT
from .opendota import write_player_games

BASE = "https://www.vlr.gg"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
    return s


def _get(s: requests.Session, url: str, tries: int = 4) -> str:
    last = None
    for i in range(tries):
        r = s.get(url, timeout=60)
        if r.status_code == 200:
            return r.text
        last = r
        time.sleep(3.0 * (i + 1))
    raise RuntimeError(f"vlr.gg {last.status_code if last else '?'} for {url}")


def parse_results_page(html: str) -> list[dict]:
    """Return [{match_id, href, date, teams, event, completed}] in page order."""
    s = BeautifulSoup(html, "lxml")
    out, current_date = [], None
    for el in s.select("div.wf-label.mod-large, a.match-item"):
        if el.name == "div":
            txt = el.get_text(" ", strip=True)
            m = re.search(r"([A-Z][a-z]{2}, [A-Z][a-z]+ \d{1,2}, \d{4})", txt)
            if m:
                current_date = dt.datetime.strptime(m.group(1), "%a, %B %d, %Y").date()
            continue
        href = el.get("href", "")
        mid = re.match(r"^/(\d+)/", href)
        if not mid:
            continue
        teams = [t.get_text(" ", strip=True) for t in el.select("div.match-item-vs-team-name")]
        event = el.select_one("div.match-item-event")
        status = el.select_one("div.ml-status")
        out.append({
            "match_id": int(mid.group(1)), "href": href, "date": current_date, "teams": teams,
            "event": " ".join(event.get_text(" ", strip=True).split()) if event else None,
            "completed": (status.get_text(strip=True).lower() == "completed") if status else True,
        })
    return out


def _side_both(cell) -> float | None:
    if cell is None:
        return None
    sp = cell.select_one("span.side.mod-both") or cell
    txt = sp.get_text(strip=True)
    try:
        return float(txt)
    except ValueError:
        return None


def parse_match(html: str, match_id: int) -> list[dict]:
    s = BeautifulSoup(html, "lxml")
    teams = [t.get_text(" ", strip=True) for t in s.select("div.match-header-link-name div.wf-title-med")]
    ts = s.select_one("div.moment-tz-convert")
    date = ts.get("data-utc-ts") if ts else None
    event_el = s.select_one("a.match-header-event")
    event = " ".join(event_el.get_text(" ", strip=True).split()) if event_el else None
    notes = " ".join(n.get_text(" ", strip=True) for n in s.select("div.match-header-vs-note"))
    bo = re.search(r"Bo(\d)", notes)
    series_format = int(bo.group(1)) if bo else None
    league = (event or "").split(":")[0].strip() or None
    playoffs = int(bool(re.search(r"playoff|final|knockout|elimination", event or "", re.I)))
    rows = []
    game_number = 0
    for g in s.select("div.vm-stats-game"):
        gid = g.get("data-game-id")
        if not gid or gid == "all":
            continue
        header = g.select_one("div.vm-stats-game-header")
        if header is None:
            continue
        names = [t.get_text(" ", strip=True) for t in header.select("div.team-name")]
        scores = [sc.get_text(strip=True) for sc in header.select("div.score")]
        map_el = header.select_one("div.map")
        map_name = None
        if map_el is not None:
            first = map_el.find("span")
            if first is not None:
                map_name = first.get_text(" ", strip=True).replace("PICK", "").strip()
        dur_el = header.select_one("div.map-duration")
        game_length = None
        if dur_el is not None:
            m = re.match(r"(\d+):(\d+)", dur_el.get_text(strip=True))
            if m:
                game_length = int(m.group(1)) + int(m.group(2)) / 60.0
        try:
            sc = [int(x) for x in scores[:2]]
        except ValueError:
            sc = [None, None]
        if len(names) < 2 or not g.select("div.ovw-row"):
            continue
        game_number += 1
        tag_order: list[str] = []
        players = []
        for row in g.select("div.ovw-row"):
            name_el = row.select_one("div.ovw-player-name")
            tag_el = row.select_one("div.ovw-player-tag")
            if name_el is None:
                continue
            tag = tag_el.get_text(strip=True) if tag_el else ""
            if tag not in tag_order:
                tag_order.append(tag)
            k = _side_both(row.select_one("span.ovw-kda-stat[data-col=kills]"))
            d = _side_both(row.select_one("span.ovw-kda-stat[data-col=deaths]"))
            a = _side_both(row.select_one("span.ovw-kda-stat[data-col=assists]"))
            agent = row.select_one("div.ovw-agents img")
            players.append({"name": name_el.get_text(strip=True), "tag": tag, "kills": k, "deaths": d, "assists": a,
                            "agent": agent.get("title") if agent else None})
        if len(tag_order) < 2:
            continue
        team_of_tag = {tag_order[0]: names[0], tag_order[1]: names[1]}
        kills_by_team: dict[str, float] = {}
        for p in players:
            tm = team_of_tag.get(p["tag"])
            if tm and p["kills"] is not None:
                kills_by_team[tm] = kills_by_team.get(tm, 0.0) + p["kills"]
        for p in players:
            tm = team_of_tag.get(p["tag"])
            if tm is None or p["kills"] is None:
                continue
            opp = names[1] if tm == names[0] else names[0]
            idx = 0 if tm == names[0] else 1
            win = None
            if None not in sc:
                win = int(sc[idx] > sc[1 - idx])
            rows.append({
                "sport": "val", "source": "vlr", "game_id": f"{match_id}-{gid}", "series_id": str(match_id), "game_number": game_number,
                "date": date, "league": league, "tier": None, "patch": None, "player_name": p["name"], "player_id": None,
                "team": tm, "opponent": opp, "role": "unknown", "side": None, "champion": p["agent"], "kills": p["kills"],
                "deaths": p["deaths"], "assists": p["assists"], "headshots": None, "team_kills": kills_by_team.get(tm),
                "opp_kills": kills_by_team.get(opp), "game_length": game_length, "win": win, "playoffs": playoffs,
                "series_format": series_format, "map_name": map_name,
            })
    return rows


def fetch(pages: int = 10, start_page: int = 1, pause: float = 1.5, since: dt.date | None = None,
          known: set[str] | None = None, progress=None) -> pd.DataFrame:
    """Scrape `pages` results pages (newest first) and every completed match on them."""
    s = _session()
    rows, seen = [], set(known or set())
    for page in range(start_page, start_page + pages):
        listing = parse_results_page(_get(s, f"{BASE}/matches/results?page={page}"))
        time.sleep(pause)
        stop = False
        for item in listing:
            if since and item["date"] and item["date"] < since:
                stop = True
                break
            if not item["completed"] or str(item["match_id"]) in seen:
                continue
            try:
                rows.extend(parse_match(_get(s, BASE + item["href"]), item["match_id"]))
            except Exception as exc:  # keep going on a bad page
                if progress:
                    progress(f"match {item['match_id']} failed: {exc}")
            seen.add(str(item["match_id"]))
            time.sleep(pause)
        if progress:
            progress(f"page {page}: {len(listing)} matches, {len(rows)} player-map rows so far")
        if stop:
            break
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df


def load(conn: sqlite3.Connection, pages: int = 10, start_page: int = 1, since: dt.date | None = None, progress=None) -> dict:
    known = {r[0] for r in conn.execute("SELECT DISTINCT series_id FROM player_games WHERE sport='val' AND source='vlr'").fetchall()}
    df = fetch(pages, start_page, since=since, known=known, progress=progress)
    if df.empty:
        return {"rows": 0, "written": 0}
    n = write_player_games(conn, df.drop(columns=["series_format", "map_name"]))
    conn.commit()
    return {"rows": int(len(df)), "written": n, "matches": int(df["series_id"].nunique())}

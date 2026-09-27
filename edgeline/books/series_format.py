"""Series formats (best-of N) for upcoming matches, so 'MAPS 1-3' props are only flagged voidable
when the series can actually end early. LoL formats come from Riot's official schedule."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd
import requests

from ..config import USER_AGENT

LOL_API = "https://esports-api.lolesports.com/persisted/gw/getSchedule"
LOL_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"


LOL_LEAGUES = "https://esports-api.lolesports.com/persisted/gw/getLeagues"
FORMAT_CACHE = Path(os.environ.get("EDGELINE_DATA_DIR", "data")) / "raw" / "lol_formats.json"
FORMAT_TTL_MIN = 30


def lol_series_events(hours_back: float = 12, hours_ahead: float = 96) -> list[dict]:
    """Upcoming LoL matches from every league's schedule: [{start, codes, names, best_of}]. Riot's global
    schedule page misses regional playoffs, and its team codes differ from the book's (TLAW vs TL, DK vs DPKC),
    so the pricer matches on start time and resolved team names, not codes alone. Cached for 30 minutes."""
    try:
        if FORMAT_CACHE.exists() and (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromtimestamp(FORMAT_CACHE.stat().st_mtime, dt.timezone.utc)).total_seconds() < FORMAT_TTL_MIN * 60:
            return json.loads(FORMAT_CACHE.read_text())
    except Exception:
        pass
    headers = {"User-Agent": USER_AGENT, "x-api-key": LOL_KEY}
    now = pd.Timestamp.now(tz="UTC")
    out: list[dict] = []
    try:
        leagues = requests.get(LOL_LEAGUES, params={"hl": "en-US"}, headers=headers, timeout=30).json()["data"]["leagues"]
    except Exception:
        return []
    seen: set[str] = set()
    for lg in leagues:
        try:
            r = requests.get(LOL_API, params={"hl": "en-US", "leagueId": lg["id"]}, headers=headers, timeout=30)
            events = r.json()["data"]["schedule"]["events"]
        except Exception:
            continue
        for e in events:
            m = e.get("match") or {}
            mid = str(m.get("id") or "")
            teams = [t for t in m.get("teams", []) if t.get("code") and t.get("code") != "TBD"]
            st = pd.to_datetime(e.get("startTime"), utc=True, errors="coerce")
            count = (m.get("strategy") or {}).get("count")
            if len(teams) != 2 or pd.isna(st) or not count or mid in seen:
                continue
            if now - pd.Timedelta(hours=hours_back) <= st <= now + pd.Timedelta(hours=hours_ahead):
                seen.add(mid)
                out.append({"start": st.isoformat(), "codes": [t["code"] for t in teams], "names": [t.get("name") or t["code"] for t in teams],
                            "best_of": int(count), "league": lg.get("slug")})
    try:
        FORMAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        FORMAT_CACHE.write_text(json.dumps(out))
    except Exception:
        pass
    return out


def lol_series_formats(hours_back: float = 12, hours_ahead: float = 96) -> dict[frozenset, int]:
    """{frozenset({codeA, codeB}): bestOf}: the code-keyed view, kept for callers that only have codes."""
    return {frozenset(e["codes"]): e["best_of"] for e in lol_series_events(hours_back, hours_ahead)}


def match_format(events: list[dict], start_time, codes: list[str], names: list[str], window_minutes: int = 90) -> tuple[int | None, str | None]:
    """Best-of for a board match: an event within the window that shares a team code or a resolved team name; if
    exactly one event sits inside a tight 20-minute window it is taken on time alone. Returns (best_of, how)."""
    st = pd.to_datetime(start_time, utc=True, errors="coerce")
    if events is None or pd.isna(st):
        return None, None
    near = [e for e in events if abs((pd.Timestamp(e["start"]) - st).total_seconds()) <= window_minutes * 60]
    codes_l = {c.lower() for c in codes if c}
    names_l = {n.lower() for n in names if n}
    for e in near:
        if codes_l & {c.lower() for c in e["codes"]}:
            return e["best_of"], "code"
    for e in near:
        if names_l & {n.lower() for n in e["names"]}:
            return e["best_of"], "name"
    tight = [e for e in near if abs((pd.Timestamp(e["start"]) - st).total_seconds()) <= 20 * 60]
    if len(tight) == 1:
        return tight[0]["best_of"], "time"
    return None, None


def voidable_with_format(map_to: int, best_of: int | None) -> bool:
    """A prop through map N can void only if the series can end before map N."""
    if map_to <= 1:
        return False
    fmt = best_of or 3
    return map_to > (fmt // 2 + 1)

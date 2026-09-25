"""Series formats (best-of N) for upcoming matches, so 'MAPS 1-3' props are only flagged voidable
when the series can actually end early. LoL formats come from Riot's official schedule."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import requests

from ..config import USER_AGENT

LOL_API = "https://esports-api.lolesports.com/persisted/gw/getSchedule"
LOL_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"


def lol_series_formats(hours_back: float = 12, hours_ahead: float = 96) -> dict[frozenset, int]:
    """{frozenset({codeA, codeB}): bestOf} for LoL matches around now (all leagues, first schedule page)."""
    try:
        r = requests.get(LOL_API, params={"hl": "en-US"}, headers={"User-Agent": USER_AGENT, "x-api-key": LOL_KEY}, timeout=30)
        r.raise_for_status()
        events = r.json()["data"]["schedule"]["events"]
    except Exception:
        return {}
    now = pd.Timestamp.now(tz="UTC")
    out: dict[frozenset, int] = {}
    for e in events:
        m = e.get("match") or {}
        codes = [t.get("code") for t in m.get("teams", []) if t.get("code") and t.get("code") != "TBD"]
        st = pd.to_datetime(e.get("startTime"), utc=True, errors="coerce")
        if len(codes) != 2 or pd.isna(st):
            continue
        if now - pd.Timedelta(hours=hours_back) <= st <= now + pd.Timedelta(hours=hours_ahead):
            count = (m.get("strategy") or {}).get("count")
            if count:
                out[frozenset(codes)] = int(count)
    return out


def voidable_with_format(map_to: int, best_of: int | None) -> bool:
    """A prop through map N can void only if the series can end before map N."""
    if map_to <= 1:
        return False
    fmt = best_of or 3
    return map_to > (fmt // 2 + 1)

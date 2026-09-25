"""Normalize book-specific prop descriptions into a common schema.

A normalized line has: sport, stat ('kills', 'deaths', 'assists', 'headshots', ...),
scope (map_from, map_to), whether it's a combo (sum over several players), and
whether it's potentially voidable (the series can end before the last map).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

STAT_ALIASES = {
    "kills": "kills",
    "deaths": "deaths",
    "assists": "assists",
    "headshots": "headshots",
    "fantasy score": "fantasy",
    "fantasy points": "fantasy",
    "kills + assists": "kills_assists",
    "kills+assists": "kills_assists",
    "cs": "cs",
    "creep score": "cs",
    "last hits": "last_hits",
    "gpm": "gpm",
    "xpm": "xpm",
    "tower kills": "tower_kills",
    "damage": "damage",
    "dmg": "damage",
    "first bloods": "first_bloods",
}

_MAP_RE = re.compile(r"^\s*(?:MAP\s+(\d+)|MAPS\s+(\d+)\s*-\s*(\d+))\s+(.+?)\s*(\((?:combo|series)\))?\s*$", re.IGNORECASE)

SPORT_BY_LEAGUE = {"lol": "lol", "cs2": "cs2", "val": "val", "dota2": "dota", "cod": "cod", "r6": "r6", "rl": "rl"}


@dataclass(frozen=True)
class PropScope:
    stat: str
    map_from: int
    map_to: int
    combo: bool

    @property
    def n_maps(self) -> int:
        return self.map_to - self.map_from + 1

    @property
    def label(self) -> str:
        span = f"MAP {self.map_from}" if self.map_from == self.map_to else f"MAPS {self.map_from}-{self.map_to}"
        return f"{span} {self.stat}{' (combo)' if self.combo else ''}"


def parse_stat_type(stat_type: str) -> PropScope | None:
    """Parse strings like 'MAP 1 Kills', 'MAPS 1-3 Kills (Combo)', 'MAPS 1-2 Headshots'."""
    m = _MAP_RE.match(stat_type or "")
    if not m:
        return None
    if m.group(1):
        a = b = int(m.group(1))
    else:
        a, b = int(m.group(2)), int(m.group(3))
    stat_raw = m.group(4).strip().lower()
    stat = STAT_ALIASES.get(stat_raw, re.sub(r"[^a-z0-9]+", "_", stat_raw).strip("_"))
    return PropScope(stat=stat, map_from=a, map_to=b, combo=bool(m.group(5)))


def sport_from_league(league_name: str) -> str:
    return SPORT_BY_LEAGUE.get((league_name or "").lower(), (league_name or "").lower())


def opponent_from_game_id(external_game_id: str, team: str) -> str | None:
    """PrizePicks external game ids look like '<TeamA><TeamB><serial>' e.g. 'fnaticHEROIC46290.1666'.

    Given one team code, strip it and the trailing numeric serial to recover the other code.
    Returns None if the team code isn't a prefix or suffix of the id.
    """
    if not external_game_id or not team:
        return None
    gid = re.sub(r"[0-9.]+$", "", external_game_id)
    gid = re.sub(r"^ex-", "", gid)
    if gid.startswith(team):
        rest = gid[len(team):]
        return rest or None
    if gid.endswith(team):
        rest = gid[: -len(team)]
        return rest or None
    return None


def voidable(sport: str, scope: PropScope, series_format: int | None = None) -> bool:
    """Whether the prop can void because the series ends before the last map.

    Without a known series format, assume BO3 for 'MAPS 1-3' style props (the
    common case) and mark them voidable. 'MAP 1' and 'MAPS 1-2' always play.
    """
    if scope.map_to <= 2:
        return False
    fmt = series_format or 3
    # In a BO3, map 3 only happens 1-1. In a BO5 maps 1-3 always play; maps 4-5 may not.
    return scope.map_to > (fmt // 2 + 1)

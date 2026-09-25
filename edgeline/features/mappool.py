"""Map-pool expectation for map-based shooters (CS2, COD): expected stat over the team's likely maps.

Lines post before the veto, so the map is unknown. For each row we compute, using only prior
games, the team's map frequencies over its last 20 maps and the player's rolling mean on each
map, and take the frequency-weighted expectation (falling back to the player's overall mean on
maps with fewer than 2 prior games). The same pass yields the current expectation per player.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

TEAM_WINDOW = 20
MIN_MAP_GAMES = 2


def map_pool_expectation(df: pd.DataFrame, stat: str = "kills", map_col: str = "champion") -> tuple[np.ndarray, dict[str, float]]:
    """Returns (per-row expectation aligned with df order, current expectation per player).

    df must be sorted chronologically and contain player_name, team, game_id, `map_col`, `stat`.
    """
    team_maps: dict[str, deque] = defaultdict(lambda: deque(maxlen=TEAM_WINDOW))
    player_map: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))  # (sum, n)
    player_all: dict[str, list] = defaultdict(lambda: [0.0, 0])
    player_team: dict[str, str] = {}
    out = np.full(len(df), np.nan)
    seen_team_game: set = set()
    vals = df[stat].to_numpy(dtype=float)
    names = df["player_name"].to_numpy()
    teams = df["team"].to_numpy()
    gids = df["game_id"].to_numpy()
    maps = df[map_col].to_numpy()

    def expect(player: str, team: str) -> float:
        hist = team_maps.get(team)
        tot = player_all[player]
        if not hist or tot[1] == 0:
            return np.nan
        base = tot[0] / tot[1]
        counts = defaultdict(int)
        for m in hist:
            counts[m] += 1
        n = len(hist)
        e = 0.0
        for m, c in counts.items():
            s, k = player_map[player].get(m, (0.0, 0))
            rate = s / k if k >= MIN_MAP_GAMES else base
            e += (c / n) * rate
        return e

    for i in range(len(df)):
        p, t, g, m, v = names[i], teams[i], gids[i], maps[i], vals[i]
        if isinstance(t, str) and t:
            out[i] = expect(p, t)
        # update after computing (as-of)
        if isinstance(m, str) and m and not np.isnan(v):
            pm = player_map[p][m]
            pm[0] += v
            pm[1] += 1
            player_all[p][0] += v
            player_all[p][1] += 1
            if isinstance(t, str) and t and (t, g) not in seen_team_game:
                team_maps[t].append(m)
                seen_team_game.add((t, g))
        if isinstance(t, str) and t:
            player_team[p] = t
    current = {p: expect(p, player_team[p]) for p in player_team}
    return out, current

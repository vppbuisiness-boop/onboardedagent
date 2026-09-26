"""Kalshi esports markets (map, game and series winners) priced with the team winner model.

Public read endpoints need no key. The exchange lists CS2, Valorant, LoL, Dota and COD map/game winners under
series tickers such as KXCS2MAP and KXLOLGAME; each event has one market per team ("FORZE Reload wins map 2").
Every scan records the best yes bid/ask, volume and our probability in `kalshi_quotes`, so the comparison with the
market accumulates even while liquidity is thin. Orders need the RSA private key paired with the API key id.
"""
from __future__ import annotations

import datetime as dt
import re
import sqlite3

import numpy as np
import pandas as pd
import requests

from ..config import USER_AGENT
from ..features.build import _prep, team_current
from ..models.predict import NameResolver
from ..models.winner import TEAM_FEATURES, fit_final, series_win_probability, team_frame

API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = {"cs2": ["KXCS2MAP", "KXCS2GAME", "KXCS2MATCH", "KXCS2MATCHWINNER", "KXCS2SERIES"], "val": ["KXVALORANTMAP", "KXVALORANTGAME", "KXVALORANTMATCH"],
          "lol": ["KXLOLMAP", "KXLOLGAME", "KXLOLMATCH", "KXLOLSERIES"], "dota": ["KXDOTA2MAP", "KXDOTA2GAME", "KXDOTA2MATCH"], "cod": ["KXCODMAP", "KXCODGAME", "KXCODMATCH"]}
TITLE_RE = re.compile(r"^(?P<team>.+?) wins (?:(?:map|game) (?P<map>\d+)|the (?P<series>match|series))", re.I)

SCHEMA = """
CREATE TABLE IF NOT EXISTS kalshi_quotes (
    ticker TEXT NOT NULL, fetched_at TEXT NOT NULL, event_ticker TEXT, sport TEXT, team TEXT, opponent TEXT, map_index INTEGER,
    market_kind TEXT, close_time TEXT, yes_bid REAL, yes_ask REAL, volume REAL, our_p REAL, PRIMARY KEY (ticker, fetched_at)
);
"""


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s


def open_markets(s: requests.Session, sport: str) -> list[dict]:
    out = []
    for tk in SERIES.get(sport, []):
        cursor = None
        for _ in range(10):
            params = {"series_ticker": tk, "status": "open", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            try:
                j = s.get(f"{API}/markets", params=params, timeout=30).json()
            except Exception:
                break
            ms = j.get("markets", [])
            out.extend(ms)
            cursor = j.get("cursor")
            if not cursor or not ms:
                break
    return out


def best_quotes(s: requests.Session, ticker: str) -> tuple[float | None, float | None]:
    """(best yes bid, best yes ask) in dollars from the order book; the yes ask is 1 - best no bid."""
    try:
        ob = s.get(f"{API}/markets/{ticker}/orderbook", timeout=30).json().get("orderbook_fp") or {}
    except Exception:
        return None, None
    yes = [float(p) for p, _ in (ob.get("yes_dollars") or []) if float(p) > 0]
    no = [float(p) for p, _ in (ob.get("no_dollars") or []) if float(p) > 0]
    return (max(yes) if yes else None), (round(1.0 - max(no), 4) if no else None)


class WinnerPricer:
    def __init__(self, conn: sqlite3.Connection, sport: str):
        pg = pd.read_sql_query("SELECT * FROM player_games WHERE sport=?", conn, params=(sport,))
        self.sport = sport
        self.t = team_frame(pg)
        self.model = fit_final(self.t)
        df = _prep(pg)
        self.state = team_current(df)
        self.last_date = df.groupby("team")["date"].max()
        self.resolver = NameResolver([], list(self.state.index))
        self.h2h: dict[tuple, list] = {}
        for r in self.t.itertuples(index=False):
            k = (r.team, r.opponent)
            self.h2h.setdefault(k, [0, 0])
            self.h2h[k][0] += int(r.win == 1); self.h2h[k][1] += 1

    def features(self, team: str, opp: str, game_number: int, asof: pd.Timestamp) -> dict | None:
        if team not in self.state.index or opp not in self.state.index:
            return None
        a, b = self.state.loc[team], self.state.loc[opp]
        h = self.h2h.get((team, opp), [0, 0])
        rest = (asof - self.last_date.get(team)).total_seconds() / 86400.0 if team in self.last_date.index else np.nan
        return {"t_elo": a["t_elo"], "o_elo": b["t_elo"], "elo_diff": a["t_elo"] - b["t_elo"], "t_win10": a["t_win10"], "o_win10": b["t_win10"],
                "matchup_win_diff": a["t_win10"] - b["t_win10"], "t_kills_mean10": a["t_kills_mean10"], "o_kills_mean10": b["t_kills_mean10"],
                "t_oppkills_mean10": a["t_oppkills_mean10"], "o_conceded_mean10": b["t_oppkills_mean10"], "t_games": a["t_games"], "o_games": b["t_games"],
                "rest_days": rest, "h2h_wins": h[0], "h2h_games": h[1], "game_number": game_number}

    def p_map(self, team: str, opp: str, game_number: int, asof: pd.Timestamp) -> float | None:
        f = self.features(team, opp, game_number, asof)
        if f is None:
            return None
        x = np.array([[f[c] for c in TEAM_FEATURES]], dtype=float)
        return float(np.clip(self.model.predict(x)[0], 0.02, 0.98))


def scan(conn: sqlite3.Connection, sports: list[str], best_of: int = 3, progress=None) -> pd.DataFrame:
    conn.executescript(SCHEMA)
    s = _session()
    now = dt.datetime.now(dt.timezone.utc)
    fetched = now.isoformat(timespec="seconds")
    rows = []
    for sport in sports:
        markets = open_markets(s, sport)
        if not markets:
            continue
        pricer = WinnerPricer(conn, sport)
        # both teams of an event come from its two markets' titles
        by_event: dict[str, list[dict]] = {}
        for m in markets:
            by_event.setdefault(m.get("event_ticker") or m["ticker"].rsplit("-", 1)[0], []).append(m)
        for ev, ms in by_event.items():
            parsed = []
            for m in ms:
                mt = TITLE_RE.match(m.get("title") or "")
                if mt:
                    parsed.append((m, mt.group("team").strip(), int(mt.group("map")) if mt.group("map") else None, bool(mt.group("series"))))
            names = [p[1] for p in parsed]
            for m, team_raw, map_idx, is_series in parsed:
                opp_raw = next((n for n in names if n != team_raw), None)
                team, opp = pricer.resolver.team(team_raw), pricer.resolver.team(opp_raw) if opp_raw else None
                p = None
                if team and opp:
                    pm = pricer.p_map(team, opp, map_idx or 1, now)
                    p = series_win_probability(pm, best_of) if (is_series and pm is not None) else pm
                bid, ask = best_quotes(s, m["ticker"])
                rows.append({"ticker": m["ticker"], "fetched_at": fetched, "event_ticker": ev, "sport": sport, "team": team or team_raw, "opponent": opp or opp_raw,
                             "map_index": map_idx, "market_kind": "series" if is_series else "map", "close_time": m.get("close_time"), "yes_bid": bid, "yes_ask": ask,
                             "volume": m.get("volume"), "our_p": p})
        if progress:
            progress(f"{sport}: {len(markets)} open markets, {sum(1 for r in rows if r['sport'] == sport and r['our_p'] is not None)} priced")
    df = pd.DataFrame(rows)
    if not df.empty:
        conn.executemany("INSERT OR REPLACE INTO kalshi_quotes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         [tuple(None if (isinstance(v, float) and np.isnan(v)) else v for v in r) for r in df[["ticker", "fetched_at", "event_ticker", "sport", "team", "opponent", "map_index", "market_kind", "close_time", "yes_bid", "yes_ask", "volume", "our_p"]].itertuples(index=False, name=None)])
        conn.commit()
        df["mid"] = df[["yes_bid", "yes_ask"]].mean(axis=1)
        df["two_sided"] = df["yes_bid"].notna() & df["yes_ask"].notna() & (df["yes_bid"] >= 0.10) & (df["yes_ask"] <= 0.90)
        df["edge_vs_ask"] = df["our_p"] - df["yes_ask"]  # buying YES at the ask
        df["edge_vs_bid"] = (1 - df["our_p"]) - (1 - df["yes_bid"])  # buying NO at 1 - bid
    return df

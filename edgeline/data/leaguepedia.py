"""League of Legends history from Leaguepedia's Cargo API (paced; anonymous access is rate limited).

Joins ScoreboardPlayers to ScoreboardGames so each player-game row carries game
length, team kills, patch, game number in the series and the tournament.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import time

import pandas as pd
import requests

from .opendota import write_player_games

API = "https://lol.fandom.com/api.php"
UA = "edgeline/0.1 (esports prop research; contact: vppbuisiness@gmail.com)"

FIELDS = [
    "SP.Link=player_link", "SP.Name=player_name", "SP.Team=team", "SP.TeamVs=opponent", "SP.Role=role",
    "SP.Champion=champion", "SP.Kills=kills", "SP.Deaths=deaths", "SP.Assists=assists", "SP.PlayerWin=win",
    "SP.GameId=game_id", "SP.MatchId=series_id", "SP.OverviewPage=overview", "SP.DateTime_UTC=date", "SP.Side=side",
    "SG.Gamelength_Number=game_length", "SG.Team1=team1", "SG.Team1Kills=team1_kills", "SG.Team2Kills=team2_kills",
    "SG.Patch=patch", "SG.N_GameInMatch=game_number", "SG.Tournament=tournament",
]


def _query(where: str, limit: int, offset: int, session: requests.Session) -> list[dict]:
    params = {
        "action": "cargoquery",
        "tables": "ScoreboardPlayers=SP,ScoreboardGames=SG",
        "join_on": "SP.GameId=SG.GameId",
        "fields": ",".join(FIELDS),
        "where": where,
        "order_by": "SP.DateTime_UTC ASC, SP.GameId ASC",
        "limit": limit,
        "offset": offset,
        "format": "json",
    }
    r = session.get(API, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()
    if "error" in payload:
        raise RuntimeError(f"Leaguepedia error: {payload['error'].get('code')}: {payload['error'].get('info')}")
    return [row["title"] for row in payload.get("cargoquery", [])]


def fetch_rows(start: dt.date, end: dt.date | None = None, pause: float = 2.5, limit: int = 500, max_pages: int = 2000) -> pd.DataFrame:
    end = end or (dt.date.today() + dt.timedelta(days=1))
    where = f"SP.DateTime_UTC >= '{start.isoformat()} 00:00:00' AND SP.DateTime_UTC < '{end.isoformat()} 00:00:00'"
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    rows, offset = [], 0
    for _ in range(max_pages):
        batch = _query(where, limit, offset, s)
        rows.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
        time.sleep(pause)
    return pd.DataFrame(rows)


def to_player_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    num = ["kills", "deaths", "assists", "game_length", "team1_kills", "team2_kills", "game_number"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    is_t1 = df["team"] == df["team1"]
    df["team_kills"] = df["team1_kills"].where(is_t1, df["team2_kills"])
    df["opp_kills"] = df["team2_kills"].where(is_t1, df["team1_kills"])
    df["league"] = df["overview"].astype(str).str.split("/").str[0]
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df["playoffs"] = df["overview"].astype(str).str.contains("Playoff", case=False).astype(int)
    out = pd.DataFrame(
        {
            "sport": "lol",
            "source": "leaguepedia",
            "game_id": df["game_id"],
            "series_id": df["series_id"],
            "game_number": df["game_number"].fillna(1).astype(int),
            "date": df["date"],
            "league": df["league"],
            "tier": None,
            "patch": df["patch"],
            "player_name": df["player_name"],
            "player_id": df["player_link"],
            "team": df["team"],
            "opponent": df["opponent"],
            "role": df["role"].astype(str).str.lower(),
            "side": df["side"].astype(str).str.lower(),
            "champion": df["champion"],
            "kills": df["kills"],
            "deaths": df["deaths"],
            "assists": df["assists"],
            "headshots": None,
            "team_kills": df["team_kills"],
            "opp_kills": df["opp_kills"],
            "game_length": df["game_length"],
            "win": pd.to_numeric(df["win"].map({"Yes": 1, "No": 0, "1": 1, "0": 0}), errors="coerce"),
            "playoffs": df["playoffs"],
        }
    )
    return out.dropna(subset=["date", "kills"])


def load(conn: sqlite3.Connection, since: dt.date, until: dt.date | None = None) -> dict:
    raw = fetch_rows(since, until)
    pg = to_player_games(raw)
    n = write_player_games(conn, pg)
    return {"raw_rows": int(len(raw)), "written": n}

"""Dota 2 professional match history via OpenDota's public SQL explorer.

One query returns thousands of player-match rows joined to league tier, teams,
series ids and notable player names. Paged by match_id descending.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import time

import pandas as pd
import requests

from ..config import USER_AGENT

EXPLORER = "https://api.opendota.com/api/explorer"
SERIES_FORMAT = {0: 1, 1: 3, 2: 5}

SQL = """
select m.match_id, m.start_time, m.duration, m.leagueid, l.name as league_name, l.tier, m.series_id, m.series_type,
       m.radiant_win, m.radiant_team_id, m.dire_team_id, tr.name as radiant_name, td.name as dire_name,
       m.radiant_score, m.dire_score,
       pm.account_id, pm.player_slot, pm.hero_id, pm.kills, pm.deaths, pm.assists, pm.lane_role, np.name as player_name
from matches m
join player_matches pm on pm.match_id = m.match_id
left join leagues l on l.leagueid = m.leagueid
left join teams tr on tr.team_id = m.radiant_team_id
left join teams td on td.team_id = m.dire_team_id
left join notable_players np on np.account_id = pm.account_id
where m.start_time >= {start_ts} and m.match_id < {before_id} and l.tier = 'professional'
order by m.match_id desc
limit {limit}
"""


def fetch_rows(start: dt.date, limit_per_page: int = 6000, max_pages: int = 200, pause: float = 1.0) -> pd.DataFrame:
    start_ts = int(dt.datetime.combine(start, dt.time(), tzinfo=dt.timezone.utc).timestamp())
    before_id = 10**12
    frames = []
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    for _ in range(max_pages):
        sql = SQL.format(start_ts=start_ts, before_id=before_id, limit=limit_per_page)
        r = s.get(EXPLORER, params={"sql": sql}, timeout=180)
        r.raise_for_status()
        rows = r.json().get("rows", [])
        if not rows:
            break
        df = pd.DataFrame(rows)
        frames.append(df)
        min_id = int(df["match_id"].min())
        # Drop the partially-fetched last match so it is re-fetched whole on the next page.
        if len(rows) >= limit_per_page:
            frames[-1] = df[df["match_id"] != min_id]
            before_id = min_id + 1
        else:
            break
        time.sleep(pause)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(["match_id", "player_slot"])


def to_player_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["is_radiant"] = df["player_slot"] < 128
    df["team"] = df["radiant_name"].where(df["is_radiant"], df["dire_name"])
    df["opponent"] = df["dire_name"].where(df["is_radiant"], df["radiant_name"])
    df["team_kills"] = df["radiant_score"].where(df["is_radiant"], df["dire_score"])
    df["opp_kills"] = df["dire_score"].where(df["is_radiant"], df["radiant_score"])
    df["win"] = (df["radiant_win"] == df["is_radiant"]).astype(int)
    df["date"] = pd.to_datetime(df["start_time"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df = df[df["account_id"].notna()].copy()  # anonymous players cannot be tracked
    df["player_name"] = df["player_name"].fillna("acct:" + df["account_id"].astype("int64").astype(str))
    # A match without a series id (BO1 or unknown) is its own series.
    df["series_id"] = df["series_id"].where(df["series_id"].notna(), df["match_id"]).astype("int64").astype(str)
    # Game number within a series by start_time order.
    order = df[["match_id", "series_id", "start_time"]].drop_duplicates("match_id").sort_values(["series_id", "start_time", "match_id"])
    order["game_number"] = order.groupby("series_id").cumcount() + 1
    df = df.merge(order[["match_id", "game_number"]], on="match_id", how="left")
    df["game_number"] = df["game_number"].fillna(1)
    df["series_format"] = df["series_type"].map(SERIES_FORMAT)
    out = pd.DataFrame(
        {
            "sport": "dota",
            "source": "opendota",
            "game_id": df["match_id"].astype(str),
            "series_id": df["series_id"],
            "game_number": df["game_number"].astype(int),
            "date": df["date"],
            "league": df["league_name"],
            "tier": df["tier"],
            "patch": None,
            "player_name": df["player_name"],
            "player_id": df["account_id"].astype("int64").astype(str),
            "team": df["team"],
            "opponent": df["opponent"],
            "role": df["lane_role"].map({1: "safe", 2: "mid", 3: "off", 4: "jungle"}).fillna("unknown") if "lane_role" in df else "unknown",
            "side": df["is_radiant"].map({True: "radiant", False: "dire"}),
            "champion": df["hero_id"].astype(str),
            "kills": df["kills"],
            "deaths": df["deaths"],
            "assists": df["assists"],
            "headshots": None,
            "team_kills": df["team_kills"],
            "opp_kills": df["opp_kills"],
            "game_length": df["duration"] / 60.0,
            "win": df["win"],
            "playoffs": None,
        }
    )
    return out


def write_player_games(conn: sqlite3.Connection, pg: pd.DataFrame) -> int:
    if pg.empty:
        return 0
    cols = list(pg.columns)
    placeholders = ",".join("?" for _ in cols)
    sql = f"INSERT OR REPLACE INTO player_games({','.join(cols)}) VALUES ({placeholders})"
    rows = [tuple(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in rec) for rec in pg.itertuples(index=False, name=None)]
    conn.executemany(sql, rows)
    return len(rows)


def load(conn: sqlite3.Connection, since: dt.date) -> dict:
    raw = fetch_rows(since)
    pg = to_player_games(raw)
    n = write_player_games(conn, pg)
    return {"raw_rows": int(len(raw)), "written": n, "matches": int(raw["match_id"].nunique()) if len(raw) else 0}

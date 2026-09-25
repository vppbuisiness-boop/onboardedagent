"""Load Oracle's Elixir yearly match CSVs (LoL) from local files.

Download from https://oracleselixir.com/tools/downloads (Google Drive folder). The
files exceed Drive's anonymous download quota at times, so this loader only reads
files you have already placed in data/raw/.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from ..config import RAW_DIR
from .opendota import write_player_games

COLS = [
    "gameid", "date", "league", "year", "split", "playoffs", "game", "patch", "side", "position", "playername",
    "playerid", "teamname", "champion", "gamelength", "result", "kills", "deaths", "assists", "teamkills", "teamdeaths",
]


def to_player_games(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[c for c in COLS if c in df.columns]].copy()
    df = df[df["position"].astype(str).str.lower() != "team"]
    # Opponent: the other team in the same gameid.
    teams = df.groupby("gameid")["teamname"].agg(lambda s: sorted(set(s.dropna())))
    df["opponent"] = [
        next((t for t in teams.get(g, []) if t != tn), None) for g, tn in zip(df["gameid"], df["teamname"])
    ]
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    day = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    pair = df.apply(lambda r: "|".join(sorted([str(r["teamname"]), str(r["opponent"])])), axis=1)
    df["series_id"] = day + ":" + df["league"].astype(str) + ":" + pair
    out = pd.DataFrame(
        {
            "sport": "lol",
            "source": "oracleselixir",
            "game_id": df["gameid"],
            "series_id": df["series_id"],
            "game_number": pd.to_numeric(df["game"], errors="coerce").fillna(1).astype(int),
            "date": df["date"],
            "league": df["league"],
            "tier": None,
            "patch": df["patch"].astype(str),
            "player_name": df["playername"],
            "player_id": df["playerid"],
            "team": df["teamname"],
            "opponent": df["opponent"],
            "role": df["position"].astype(str).str.lower(),
            "side": df["side"].astype(str).str.lower(),
            "champion": df["champion"],
            "kills": df["kills"],
            "deaths": df["deaths"],
            "assists": df["assists"],
            "headshots": None,
            "team_kills": df["teamkills"],
            "opp_kills": df["teamdeaths"],
            "game_length": pd.to_numeric(df["gamelength"], errors="coerce") / 60.0,
            "win": pd.to_numeric(df["result"], errors="coerce"),
            "playoffs": pd.to_numeric(df["playoffs"], errors="coerce"),
        }
    )
    return out.dropna(subset=["date", "kills", "player_name"])


def load_files(conn: sqlite3.Connection, paths: list[Path] | None = None) -> dict:
    paths = paths or sorted(RAW_DIR.glob("*OraclesElixir*.csv"))
    total = 0
    for p in paths:
        df = pd.read_csv(p, low_memory=False)
        if "gameid" not in df.columns:
            raise ValueError(f"{p} does not look like an Oracle's Elixir CSV (no gameid column)")
        total += write_player_games(conn, to_player_games(df))
    return {"files": [str(p) for p in paths], "written": total}

"""As-of feature engineering on player_games.

Every feature for a row is computed only from games strictly before that row
(shift-then-roll), so training rows never see their own outcome. The same
aggregations, evaluated on the full history without the shift, give the
current state used at prediction time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATS = ["kills", "deaths", "assists", "headshots"]
WINDOWS = (5, 10, 20)
EWM_HALFLIFE = 8

FEATURE_COLUMNS = (
    [f"p_{s}_ewm" for s in STATS]
    + [f"p_{s}_mean{w}" for s in STATS for w in WINDOWS]
    + [f"p_{s}_std10" for s in STATS]
    + ["p_kshare_mean10", "p_games", "p_gl_mean10", "p_days_since"]
    + ["t_kills_mean10", "t_oppkills_mean10", "t_win10", "t_gl_mean10", "t_games"]
    + ["o_kills_mean10", "o_conceded_mean10", "o_win10", "o_games"]
    + ["matchup_win_diff", "game_number", "playoffs"]
)
CATEGORICAL = ["role", "league", "tier"]


def _prep(pg: pd.DataFrame) -> pd.DataFrame:
    df = pg.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date", "player_name"])
    for c in STATS + ["team_kills", "opp_kills", "game_length", "win", "game_number", "playoffs"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    df["playoffs"] = df["playoffs"].fillna(0)
    df["game_number"] = df["game_number"].fillna(1)
    df["kshare"] = (df["kills"] / df["team_kills"].replace(0, np.nan)).clip(0, 1)
    df["role"] = df["role"].fillna("unknown").astype(str)
    df["league"] = df["league"].fillna("unknown").astype(str)
    df["tier"] = (df["tier"] if "tier" in df.columns else pd.Series(index=df.index, dtype=object)).fillna("unknown").astype(str)
    return df.sort_values(["date", "game_id"]).reset_index(drop=True)


def _team_games(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game): team kills, kills conceded, win, game length."""
    tg = (
        df.groupby(["team", "game_id"], as_index=False)
        .agg(date=("date", "first"), team_kills=("team_kills", "max"), opp_kills=("opp_kills", "max"), win=("win", "max"), game_length=("game_length", "max"))
        .sort_values(["team", "date", "game_id"])
    )
    return tg


def _prior_roll(g: pd.DataFrame, col: str, w: int, fn: str) -> pd.Series:
    s = g[col].shift(1)
    r = s.rolling(w, min_periods=1)
    return getattr(r, fn)()


def player_features(df: pd.DataFrame, shift: bool = True) -> pd.DataFrame:
    """Per-row as-of player features. With shift=False the last row of each player is the current state."""
    out = df.copy()
    grp = out.groupby("player_name", sort=False)
    sh = 1 if shift else 0
    for s in STATS:
        out[f"p_{s}_ewm"] = grp[s].transform(lambda x: x.shift(sh).ewm(halflife=EWM_HALFLIFE, min_periods=1).mean())
        for w in WINDOWS:
            out[f"p_{s}_mean{w}"] = grp[s].transform(lambda x, w=w: x.shift(sh).rolling(w, min_periods=1).mean())
        out[f"p_{s}_std10"] = grp[s].transform(lambda x: x.shift(sh).rolling(10, min_periods=2).std())
    out["p_kshare_mean10"] = grp["kshare"].transform(lambda x: x.shift(sh).rolling(10, min_periods=1).mean())
    out["p_gl_mean10"] = grp["game_length"].transform(lambda x: x.shift(sh).rolling(10, min_periods=1).mean())
    out["p_games"] = grp.cumcount() + (0 if shift else 1)
    prev_date = grp["date"].shift(sh)
    out["p_days_since"] = (out["date"] - prev_date).dt.total_seconds() / 86400.0
    return out


def team_state(df: pd.DataFrame) -> pd.DataFrame:
    """Team features keyed by (team, game_id) computed from prior team games; plus a current-state table."""
    tg = _team_games(df)
    g = tg.groupby("team", sort=False)
    tg["t_kills_mean10"] = g["team_kills"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    tg["t_oppkills_mean10"] = g["opp_kills"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    tg["t_win10"] = g["win"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    tg["t_gl_mean10"] = g["game_length"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    tg["t_games"] = g.cumcount()
    return tg


def team_current(df: pd.DataFrame) -> pd.DataFrame:
    tg = _team_games(df)
    g = tg.groupby("team", sort=False)
    cur = pd.DataFrame(
        {
            "t_kills_mean10": g["team_kills"].apply(lambda x: x.tail(10).mean()),
            "t_oppkills_mean10": g["opp_kills"].apply(lambda x: x.tail(10).mean()),
            "t_win10": g["win"].apply(lambda x: x.tail(10).mean()),
            "t_gl_mean10": g["game_length"].apply(lambda x: x.tail(10).mean()),
            "t_games": g.size(),
        }
    )
    return cur


def _attach_opponent(out: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    opp = tg.rename(
        columns={
            "team": "opponent",
            "t_kills_mean10": "o_kills_mean10",
            "t_oppkills_mean10": "o_conceded_mean10",
            "t_win10": "o_win10",
            "t_games": "o_games",
        }
    )[["opponent", "game_id", "o_kills_mean10", "o_conceded_mean10", "o_win10", "o_games"]]
    return out.merge(opp, on=["opponent", "game_id"], how="left")


def build_training_frame(pg: pd.DataFrame) -> pd.DataFrame:
    """Full as-of feature frame for model training (one row per player-game)."""
    df = _prep(pg)
    out = player_features(df, shift=True)
    tg = team_state(df)
    out = out.merge(tg[["team", "game_id", "t_kills_mean10", "t_oppkills_mean10", "t_win10", "t_gl_mean10", "t_games"]], on=["team", "game_id"], how="left")
    out = _attach_opponent(out, tg)
    out["matchup_win_diff"] = out["t_win10"] - out["o_win10"]
    return out


def current_state(pg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(player_state, team_state) as of now: latest row per player with unshifted rolling stats."""
    df = _prep(pg)
    pf = player_features(df, shift=False)
    latest = pf.groupby("player_name", sort=False).tail(1).set_index("player_name")
    latest["p_days_since"] = (pd.Timestamp.now(tz="UTC") - latest["date"]).dt.total_seconds() / 86400.0
    return latest, team_current(df)


def assemble_prediction_row(player_row: pd.Series, team_row: pd.Series | None, opp_row: pd.Series | None,
                            game_number: int, playoffs: int, league: str | None, role: str | None) -> dict:
    feat = {c: player_row.get(c, np.nan) for c in FEATURE_COLUMNS if c.startswith("p_")}
    for c in ["t_kills_mean10", "t_oppkills_mean10", "t_win10", "t_gl_mean10", "t_games"]:
        feat[c] = team_row.get(c, np.nan) if team_row is not None else np.nan
    for src, dst in [("t_kills_mean10", "o_kills_mean10"), ("t_oppkills_mean10", "o_conceded_mean10"), ("t_win10", "o_win10"), ("t_games", "o_games")]:
        feat[dst] = opp_row.get(src, np.nan) if opp_row is not None else np.nan
    feat["matchup_win_diff"] = feat["t_win10"] - feat["o_win10"] if not (pd.isna(feat["t_win10"]) or pd.isna(feat["o_win10"])) else np.nan
    feat["game_number"] = game_number
    feat["playoffs"] = playoffs
    feat["role"] = role or player_row.get("role", "unknown")
    feat["league"] = league or player_row.get("league", "unknown")
    feat["tier"] = player_row.get("tier", "unknown") or "unknown"
    return feat

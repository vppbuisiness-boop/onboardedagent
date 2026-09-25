"""As-of feature engineering on player_games.

Every feature for a row is computed only from games strictly before that row
(shift-then-roll, or a rating updated after each game), so training rows never
see their own outcome. The same aggregations evaluated on the full history
without the shift give the current state used at prediction time.

Feature groups
- p_*   player rolling form (EWM, rolling means/stds, kill share, game length, rest days)
- pr_*  player rolling form within a role (role = game mode in COD, position in MOBAs)
- t_*   team rolling form and Elo strength
- o_*   opponent rolling form, Elo, and role-matchup: what the opponent's player in the
        same role scores (drives deaths) and what the opponent concedes to that role (drives kills)
- elo_diff / elo_absdiff / p_win_elo  strength gap (win probability proxy for the missing moneyline)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATS = ["kills", "deaths", "assists", "headshots"]
WINDOWS = (5, 10, 20)
EWM_HALFLIFE = 8
ELO_K = 24.0
ELO_START = 1500.0

TEAM_COLS = ["t_kills_mean10", "t_oppkills_mean10", "t_win10", "t_gl_mean10", "t_games", "t_elo"]
OPP_COLS = ["o_kills_mean10", "o_conceded_mean10", "o_win10", "o_games", "o_elo"]
ROLE_MATCHUP_COLS = ["o_role_kills10", "o_role_conceded10"]

FEATURE_COLUMNS = (
    [f"p_{s}_ewm" for s in STATS]
    + [f"p_{s}_mean{w}" for s in STATS for w in WINDOWS]
    + [f"p_{s}_std10" for s in STATS]
    + ["p_kshare_mean10", "p_games", "p_gl_mean10", "p_days_since"]
    + TEAM_COLS
    + OPP_COLS
    + ["matchup_win_diff", "elo_diff", "elo_absdiff", "p_win_elo", "game_number", "playoffs"]
    + [f"pr_{s}_mean10" for s in STATS] + ["pr_games"]
    + ROLE_MATCHUP_COLS
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
    """One row per (team, game): team kills, kills conceded, win, game length, opponent."""
    tg = (
        df.groupby(["team", "game_id"], as_index=False)
        .agg(date=("date", "first"), opponent=("opponent", "first"), team_kills=("team_kills", "max"), opp_kills=("opp_kills", "max"),
             win=("win", "max"), game_length=("game_length", "max"))
        .sort_values(["team", "date", "game_id"])
    )
    return tg


# ---------------------------------------------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------------------------------------------
def elo_ratings(tg: pd.DataFrame, k: float = ELO_K) -> tuple[pd.DataFrame, dict[str, float]]:
    """Pre-game Elo per (team, game_id) from map results in chronological order, plus final ratings.

    Each game appears once per team in tg; we process each game_id once using the row whose team sorts first.
    """
    games = tg.dropna(subset=["opponent"]).sort_values(["date", "game_id"])
    games = games[games["team"] < games["opponent"]].drop_duplicates("game_id") if (games["team"] < games["opponent"]).any() else games.drop_duplicates("game_id")
    rating: dict[str, float] = {}
    pre_rows = []
    for g in games.itertuples(index=False):
        a, b = g.team, g.opponent
        ra, rb = rating.get(a, ELO_START), rating.get(b, ELO_START)
        pre_rows.append((a, g.game_id, ra))
        pre_rows.append((b, g.game_id, rb))
        if pd.isna(g.win):
            continue
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        sa = float(g.win)
        rating[a] = ra + k * (sa - ea)
        rating[b] = rb + k * ((1 - sa) - (1 - ea))
    pre = pd.DataFrame(pre_rows, columns=["team", "game_id", "t_elo"])
    return pre, rating


# ---------------------------------------------------------------------------------------------------------------
# Player and role form
# ---------------------------------------------------------------------------------------------------------------
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


def player_role_features(df: pd.DataFrame, shift: bool = True) -> pd.DataFrame:
    out = df.copy()
    grp = out.groupby(["player_name", "role"], sort=False)
    sh = 1 if shift else 0
    for s in STATS:
        out[f"pr_{s}_mean10"] = grp[s].transform(lambda x: x.shift(sh).rolling(10, min_periods=1).mean())
    out["pr_games"] = grp.cumcount() + (0 if shift else 1)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Team form, role matchups
# ---------------------------------------------------------------------------------------------------------------
def team_state(df: pd.DataFrame) -> pd.DataFrame:
    """Team features keyed by (team, game_id) computed from prior team games (as-of), including pre-game Elo."""
    tg = _team_games(df)
    g = tg.groupby("team", sort=False)
    tg["t_kills_mean10"] = g["team_kills"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    tg["t_oppkills_mean10"] = g["opp_kills"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    tg["t_win10"] = g["win"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    tg["t_gl_mean10"] = g["game_length"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    tg["t_games"] = g.cumcount()
    pre, _ = elo_ratings(tg)
    tg = tg.merge(pre, on=["team", "game_id"], how="left")
    tg["t_elo"] = tg["t_elo"].fillna(ELO_START)
    return tg


def role_matchup_tables(df: pd.DataFrame, shift: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """As-of rolling stats per (team, role):
    kills   = what the team's own players in that role score (mean per player-game in the role)
    conceded = what the team concedes to the opponent's players in that role."""
    own = df.groupby(["team", "game_id", "role"], as_index=False).agg(date=("date", "first"), kills=("kills", "mean"))
    own = own.sort_values(["team", "role", "date", "game_id"])
    sh = 1 if shift else 0
    own["tr_kills10"] = own.groupby(["team", "role"], sort=False)["kills"].transform(lambda x: x.shift(sh).rolling(10, min_periods=1).mean())
    conceded = df.dropna(subset=["opponent"]).groupby(["opponent", "game_id", "role"], as_index=False).agg(date=("date", "first"), conceded=("kills", "mean"))
    conceded = conceded.rename(columns={"opponent": "team"}).sort_values(["team", "role", "date", "game_id"])
    conceded["tr_conceded10"] = conceded.groupby(["team", "role"], sort=False)["conceded"].transform(lambda x: x.shift(sh).rolling(10, min_periods=1).mean())
    return own[["team", "game_id", "role", "tr_kills10"]], conceded[["team", "game_id", "role", "tr_conceded10"]]


def team_current(df: pd.DataFrame) -> pd.DataFrame:
    tg = _team_games(df)
    g = tg.groupby("team", sort=False)
    _, final_elo = elo_ratings(tg)
    cur = pd.DataFrame(
        {
            "t_kills_mean10": g["team_kills"].apply(lambda x: x.tail(10).mean()),
            "t_oppkills_mean10": g["opp_kills"].apply(lambda x: x.tail(10).mean()),
            "t_win10": g["win"].apply(lambda x: x.tail(10).mean()),
            "t_gl_mean10": g["game_length"].apply(lambda x: x.tail(10).mean()),
            "t_games": g.size(),
        }
    )
    cur["t_elo"] = pd.Series(final_elo).reindex(cur.index).fillna(ELO_START)
    own, conceded = role_matchup_tables(df, shift=False)
    rk = own.groupby(["team", "role"]).tail(1)
    rc = conceded.groupby(["team", "role"]).tail(1)
    role_kills: dict[str, dict] = {}
    for r in rk.itertuples(index=False):
        role_kills.setdefault(r.team, {})[r.role] = float(r.tr_kills10)
    role_conceded: dict[str, dict] = {}
    for r in rc.itertuples(index=False):
        role_conceded.setdefault(r.team, {})[r.role] = float(r.tr_conceded10)
    cur["role_kills"] = pd.Series(role_kills).reindex(cur.index)
    cur["role_conceded"] = pd.Series(role_conceded).reindex(cur.index)
    return cur


def _attach_opponent(out: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    opp = tg[["team", "game_id"] + TEAM_COLS].rename(
        columns={
            "team": "opponent",
            "t_kills_mean10": "o_kills_mean10",
            "t_oppkills_mean10": "o_conceded_mean10",
            "t_win10": "o_win10",
            "t_games": "o_games",
            "t_elo": "o_elo",
        }
    )[["opponent", "game_id"] + OPP_COLS]
    return out.merge(opp, on=["opponent", "game_id"], how="left")


def _strength_features(out: pd.DataFrame) -> pd.DataFrame:
    out["matchup_win_diff"] = out["t_win10"] - out["o_win10"]
    out["elo_diff"] = out["t_elo"] - out["o_elo"]
    out["elo_absdiff"] = out["elo_diff"].abs()
    out["p_win_elo"] = 1.0 / (1.0 + 10 ** (-out["elo_diff"] / 400.0))
    return out


def build_training_frame(pg: pd.DataFrame) -> pd.DataFrame:
    """Full as-of feature frame for model training (one row per player-game)."""
    df = _prep(pg)
    out = player_features(df, shift=True)
    out = player_role_features(out, shift=True)
    tg = team_state(df)
    out = out.merge(tg[["team", "game_id"] + TEAM_COLS], on=["team", "game_id"], how="left")
    out = _attach_opponent(out, tg)
    own, conceded = role_matchup_tables(df, shift=True)
    out = out.merge(own.rename(columns={"team": "opponent", "tr_kills10": "o_role_kills10"}), on=["opponent", "game_id", "role"], how="left")
    out = out.merge(conceded.rename(columns={"team": "opponent", "tr_conceded10": "o_role_conceded10"}), on=["opponent", "game_id", "role"], how="left")
    return _strength_features(out)


def current_state(pg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(player_state, team_state) as of now: latest row per player with unshifted rolling stats.

    player_state carries per-role values in `role_state` ({role: {pr_*}}); team_state carries Elo and
    per-role matchup dicts (`role_kills`, `role_conceded`)."""
    df = _prep(pg)
    pf = player_features(df, shift=False)
    pf = player_role_features(pf, shift=False)
    latest = pf.groupby("player_name", sort=False).tail(1).set_index("player_name")
    latest["p_days_since"] = (pd.Timestamp.now(tz="UTC") - latest["date"]).dt.total_seconds() / 86400.0
    cols = [f"pr_{s}_mean10" for s in STATS] + ["pr_games"]
    per_role = pf.groupby(["player_name", "role"], sort=False).tail(1)
    role_state: dict[str, dict] = {}
    for r in per_role.itertuples(index=False):
        role_state.setdefault(r.player_name, {})[r.role] = {c: getattr(r, c) for c in cols}
    latest["role_state"] = pd.Series(role_state)
    return latest, team_current(df)


def assemble_prediction_row(player_row: pd.Series, team_row: pd.Series | None, opp_row: pd.Series | None,
                            game_number: int, playoffs: int, league: str | None, role: str | None) -> dict:
    feat = {c: player_row.get(c, np.nan) for c in FEATURE_COLUMNS if c.startswith("p_") and c != "p_win_elo"}
    for c in TEAM_COLS:
        feat[c] = team_row.get(c, np.nan) if team_row is not None else np.nan
    for src, dst in [("t_kills_mean10", "o_kills_mean10"), ("t_oppkills_mean10", "o_conceded_mean10"), ("t_win10", "o_win10"), ("t_games", "o_games"), ("t_elo", "o_elo")]:
        feat[dst] = opp_row.get(src, np.nan) if opp_row is not None else np.nan
    feat["matchup_win_diff"] = feat["t_win10"] - feat["o_win10"] if not (pd.isna(feat["t_win10"]) or pd.isna(feat["o_win10"])) else np.nan
    if pd.isna(feat["t_elo"]) or pd.isna(feat["o_elo"]):
        feat["elo_diff"] = feat["elo_absdiff"] = feat["p_win_elo"] = np.nan
    else:
        feat["elo_diff"] = feat["t_elo"] - feat["o_elo"]
        feat["elo_absdiff"] = abs(feat["elo_diff"])
        feat["p_win_elo"] = 1.0 / (1.0 + 10 ** (-feat["elo_diff"] / 400.0))
    feat["game_number"] = game_number
    feat["playoffs"] = playoffs
    feat["role"] = role or player_row.get("role", "unknown")
    feat["league"] = league or player_row.get("league", "unknown")
    feat["tier"] = player_row.get("tier", "unknown") or "unknown"
    rs = player_row.get("role_state") if isinstance(player_row.get("role_state"), dict) else {}
    rr = rs.get(feat["role"]) or {}
    for c in [f"pr_{s}_mean10" for s in STATS] + ["pr_games"]:
        feat[c] = rr.get(c, np.nan)
    ok = opp_row.get("role_kills") if opp_row is not None else None
    oc = opp_row.get("role_conceded") if opp_row is not None else None
    feat["o_role_kills10"] = (ok or {}).get(feat["role"], np.nan) if isinstance(ok, dict) else np.nan
    feat["o_role_conceded10"] = (oc or {}).get(feat["role"], np.nan) if isinstance(oc, dict) else np.nan
    return feat

"""Map-winner model per sport, walk-forward validated, plus series probabilities.

Team rows come from the same as-of feature frame the prop models use (one row per team per map, aggregated from
its players): pre-game Elo for both sides, recent win rate, kills pace, rest days and a head-to-head record. A
LightGBM classifier is compared with a logistic fit on the Elo gap alone; the point is to know whether the
extra features add calibrated information before pricing exchange markets with it.
"""
from __future__ import annotations

import warnings
from itertools import combinations
from math import comb

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from ..features.build import build_training_frame

TEAM_FEATURES = ["t_elo", "o_elo", "elo_diff", "t_win10", "o_win10", "matchup_win_diff", "t_kills_mean10", "o_kills_mean10",
                 "t_oppkills_mean10", "o_conceded_mean10", "t_games", "o_games", "rest_days", "h2h_wins", "h2h_games", "game_number"]
PARAMS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 200, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1, "seed": 7}


def team_frame(pg: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team) with as-of features and the map result."""
    f = build_training_frame(pg)
    f = f.dropna(subset=["team", "opponent", "win"])
    keep = ["sport", "game_id", "series_id", "game_number", "date", "team", "opponent", "win", "t_elo", "o_elo", "elo_diff", "t_win10", "o_win10",
            "matchup_win_diff", "t_kills_mean10", "o_kills_mean10", "t_oppkills_mean10", "o_conceded_mean10", "t_games", "o_games"]
    t = f[keep].groupby(["game_id", "team"], as_index=False).first().sort_values(["date", "game_id"]).reset_index(drop=True)
    # rest days and head-to-head, as-of
    t["rest_days"] = t.groupby("team")["date"].diff().dt.total_seconds() / 86400.0
    h2h_w, h2h_n = [], []
    rec: dict[tuple, list] = {}
    for r in t.itertuples(index=False):
        k = (r.team, r.opponent)
        w, n = rec.get(k, [0, 0])
        h2h_w.append(w); h2h_n.append(n)
        rec.setdefault(k, [0, 0]); rec[k][0] += int(r.win == 1); rec[k][1] += 1
    t["h2h_wins"], t["h2h_games"] = h2h_w, h2h_n
    return t


def walk_forward(t: pd.DataFrame, months: int = 5, min_games: int = 5, progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    warnings.filterwarnings("ignore")
    df = t[(t["t_games"] >= min_games) & (t["o_games"] >= min_games)].copy()
    last = df["date"].max()
    starts = [(last.normalize().replace(day=1) - pd.DateOffset(months=k)) for k in range(months - 1, -1, -1)]
    rows, preds = [], []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else last + pd.Timedelta(days=1)
        past = df[df["date"] < start]; fold = df[(df["date"] >= start) & (df["date"] < end)]
        if len(past) < 1000 or len(fold) < 100:
            continue
        Xp, yp = past[TEAM_FEATURES].to_numpy(dtype=float), past["win"].to_numpy(dtype=int)
        Xf, yf = fold[TEAM_FEATURES].to_numpy(dtype=float), fold["win"].to_numpy(dtype=int)
        booster = lgb.train(PARAMS, lgb.Dataset(Xp, label=yp), num_boost_round=400)
        p_gbm = np.clip(booster.predict(Xf), 0.01, 0.99)
        elo = LogisticRegression().fit(past[["elo_diff"]].fillna(0).to_numpy(), yp)
        p_elo = np.clip(elo.predict_proba(fold[["elo_diff"]].fillna(0).to_numpy())[:, 1], 0.01, 0.99)
        rows.append({"fold": start.strftime("%Y-%m"), "games": int(len(fold) // 2), "logloss_gbm": log_loss(yf, p_gbm), "logloss_elo": log_loss(yf, p_elo),
                     "brier_gbm": brier_score_loss(yf, p_gbm), "brier_elo": brier_score_loss(yf, p_elo),
                     "acc_gbm": float(((p_gbm > 0.5) == yf).mean()), "acc_elo": float(((p_elo > 0.5) == yf).mean())})
        preds.append(pd.DataFrame({"fold": start.strftime("%Y-%m"), "game_id": fold["game_id"].to_numpy(), "team": fold["team"].to_numpy(), "y": yf, "p_gbm": p_gbm, "p_elo": p_elo}))
        if progress:
            progress(f"fold {start:%Y-%m}: {len(fold) // 2} maps")
    return pd.DataFrame(rows), (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame())


def reliability(pred: pd.DataFrame, col: str = "p_gbm", bins=(0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0)) -> pd.DataFrame:
    b = pd.cut(pred[col], bins)
    return pred.groupby(b, observed=True).agg(n=("y", "size"), predicted=(col, "mean"), observed=("y", "mean")).reset_index()


def series_win_probability(p_map: float, best_of: int) -> float:
    """P(win a best-of-n) from a constant per-map probability, maps independent."""
    need = best_of // 2 + 1
    return float(sum(comb(need - 1 + k, k) * p_map**need * (1 - p_map) ** k for k in range(0, best_of - need + 1)))


def fit_final(t: pd.DataFrame, min_games: int = 5) -> lgb.Booster:
    df = t[(t["t_games"] >= min_games) & (t["o_games"] >= min_games)]
    return lgb.train(PARAMS, lgb.Dataset(df[TEAM_FEATURES].to_numpy(dtype=float), label=df["win"].to_numpy(dtype=int)), num_boost_round=400)

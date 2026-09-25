"""Walk-forward historical backtest against synthetic line-setters.

There is no public archive of historical PrizePicks esports lines, so a historical ROI can only
be estimated against modeled lines. This module does that as honestly as the data allows:

- walk-forward: for each monthly fold the model is trained only on games before the fold, and the
  fold's games are priced with features that use only prior games (the normal as-of frame);
- two synthetic books set the lines for every player-game in the fold:
    naive     line = the player's trailing 10-game mean in the role, rounded to .5
    booklike  line = a separate LightGBM model fit on the same past data using only the basic
              averages a small book would use (player last-10/20 per role, opponent kills conceded,
              team pace, role, map number), rounded to .5
- the live pricing rules are applied exactly: 25% shrink toward the line, NB tail + calibrator,
  bet the side with probability >= 60%;
- output per fold and pooled: picks, hit rate, Wilson interval, implied 4-pick POWER ROI.

Read the result as "edge over a book that prices like this", not as realized ROI on PrizePicks.
Real books are at least as sharp as the booklike setter; the forward test on captured lines is
the only real ROI.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..features.build import CATEGORICAL
from ..grading.roi import parlay_roi, wilson
from .distributions import over_under_push
from .props import LGB_PARAMS, train

BOOK_FEATURES = ["p_kills_mean10", "p_kills_mean20", "pr_kills_mean10", "pr_games", "o_conceded_mean10", "t_kills_mean10",
                 "t_oppkills_mean10", "o_role_conceded10", "game_number"]


def _book_features(stat: str) -> list[str]:
    return [f.replace("kills", stat) if f.startswith(("p_", "pr_")) else f for f in BOOK_FEATURES]


def fit_booklike(train_df: pd.DataFrame, stat: str, cat_levels: dict) -> lgb.Booster:
    feats = _book_features(stat) + ["role"]
    X = train_df.reindex(columns=feats).copy()
    X["role"] = pd.Categorical(X["role"].astype(str), categories=cat_levels["role"])
    y = train_df[stat].to_numpy(dtype=float)
    params = {**LGB_PARAMS, "num_leaves": 15, "min_data_in_leaf": 100, "verbose": -1}
    return lgb.train(params, lgb.Dataset(X, label=y, categorical_feature=["role"]), num_boost_round=300)


def predict_booklike(booster: lgb.Booster, df: pd.DataFrame, stat: str, cat_levels: dict) -> np.ndarray:
    feats = _book_features(stat) + ["role"]
    X = df.reindex(columns=feats).copy()
    X["role"] = pd.Categorical(X["role"].astype(str), categories=cat_levels["role"])
    return np.clip(booster.predict(X), 0.05, None)


def _to_line(mean: np.ndarray) -> np.ndarray:
    return np.floor(mean) + 0.5


def price_fold(model, fold: pd.DataFrame, stat: str, lines: np.ndarray, shrink: float, threshold: float) -> pd.DataFrame:
    mu = model.predict_mu(fold)
    mu_adj = (1 - shrink) * mu + shrink * lines
    y = fold[stat].to_numpy(dtype=float)
    p_over_raw = np.array([over_under_push(l, m, model.r)[0] for l, m in zip(lines, mu_adj)])
    p_over = np.asarray(model.calibrate(p_over_raw))
    p_under = 1 - p_over
    lean_over = p_over >= p_under
    prob = np.where(lean_over, p_over, p_under)
    pick = prob >= threshold
    hit = np.where(lean_over, y > lines, y < lines)
    return pd.DataFrame({"date": fold["date"].to_numpy(), "line": lines, "mu": mu, "mu_adj": mu_adj, "actual": y, "prob": prob,
                         "lean_over": lean_over, "pick": pick, "hit": hit})


def walk_forward(frame: pd.DataFrame, sport: str, stat: str, months: int = 5, shrink: float = 0.25, threshold: float = 0.60,
                 min_games: int = 3, progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-fold summary, pooled picks)."""
    warnings.filterwarnings("ignore")
    df = frame.dropna(subset=[stat]).copy()
    df = df[(df[stat] >= 0) & (df["p_games"] >= min_games)].sort_values("date")
    last = df["date"].max()
    starts = [(last.normalize().replace(day=1) - pd.DateOffset(months=k)) for k in range(months - 1, -1, -1)]
    rows, picks = [], []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else last + pd.Timedelta(days=1)
        past = df[df["date"] < start]
        fold = df[(df["date"] >= start) & (df["date"] < end)]
        if len(past) < 2000 or len(fold) < 200:
            continue
        model = train(past, sport, stat, valid_frac=0.2)
        cat_levels = model.cat_levels
        fold = fold[fold["role"].astype(str).isin(cat_levels["role"]) | True]  # unseen roles handled by pandas categorical as NaN
        book = fit_booklike(past, stat, cat_levels)
        naive_mean = fold[f"pr_{stat}_mean10"].fillna(fold[f"p_{stat}_mean10"]).to_numpy(dtype=float)
        ok = ~np.isnan(naive_mean)
        fold = fold[ok]
        naive_mean = naive_mean[ok]
        book_mean = predict_booklike(book, fold, stat, cat_levels)
        for book_name, lines in (("naive", _to_line(naive_mean)), ("booklike", _to_line(book_mean))):
            res = price_fold(model, fold, stat, lines, shrink, threshold)
            res["book"], res["fold"] = book_name, start.strftime("%Y-%m")
            picks.append(res)
            sel = res[res["pick"]]
            n, h = int(len(sel)), int(sel["hit"].sum())
            p, lo, hi = wilson(h, n) if n else (float("nan"),) * 3
            rows.append({"fold": start.strftime("%Y-%m"), "book": book_name, "games": int(len(res)), "picks": n, "hits": h,
                         "hit_rate": p, "ci_low": lo, "ci_high": hi, "parlay4_roi": parlay_roi(p) if n else float("nan"),
                         "book_mae": float(np.mean(np.abs(res["actual"] - lines))), "model_mae": float(np.mean(np.abs(res["actual"] - res["mu"])))})
        if progress:
            progress(f"{sport}/{stat} fold {start:%Y-%m}: trained on {len(past)} rows, priced {len(fold)}")
    summary = pd.DataFrame(rows)
    pooled = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
    return summary, pooled


def pooled_summary(pooled: pd.DataFrame) -> pd.DataFrame:
    out = []
    for book, g in pooled.groupby("book"):
        sel = g[g["pick"]]
        n, h = int(len(sel)), int(sel["hit"].sum())
        p, lo, hi = wilson(h, n) if n else (float("nan"),) * 3
        out.append({"book": book, "games": int(len(g)), "picks": n, "pick_share": n / max(len(g), 1), "hits": h, "hit_rate": p,
                    "ci_low": lo, "ci_high": hi, "parlay4_roi": parlay_roi(p) if n else float("nan"),
                    "parlay4_roi_ci_low": parlay_roi(lo) if n else float("nan"), "parlay4_roi_ci_high": parlay_roi(hi) if n else float("nan")})
    return pd.DataFrame(out)

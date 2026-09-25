"""Walk-forward historical backtest against synthetic line-setters.

There is no public archive of historical PrizePicks esports lines, so a historical ROI can only
be estimated against modeled lines. This module does that as honestly as the data allows:

- walk-forward: for each monthly fold the model is trained only on games before the fold, and the
  fold's games are priced with features that use only prior games (the normal as-of frame);
- two synthetic books set the lines for every player-game in the fold:
    naive     line = the player's trailing 10-game mean in the role, set at the nearest median-fair .5
    booklike  line = a separate LightGBM model fit on the same past data using only the basic
              averages a small book would use (player last-10/20 per role, opponent kills conceded,
              team pace, role, map number), set at the nearest median-fair .5 (see fair_line)
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
from .copula import Component, sum_over_under_push
from .distributions import fair_line, over_under_push
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


def _to_line(mean: np.ndarray, r: float) -> np.ndarray:
    return fair_line(mean, r)


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
        for book_name, lines in (("naive", _to_line(naive_mean, model.r)), ("booklike", _to_line(book_mean, model.r))):
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


# ---------------------------------------------------------------------------------------------------------------
# Two-map sums ("MAPS 1-2" lines), the most common line shape on the board
# ---------------------------------------------------------------------------------------------------------------
def two_map_pairs(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Map-1 rows of every (series, player) whose series reached a map 2, with `actual_sum` = stat(map 1) + stat(map 2).

    Only the map-1 row's features are used for both maps, exactly as the live pricer must (map 2's as-of
    features would already contain map 1's result)."""
    m1 = df[df["game_number"] == 1]
    m2 = df[df["game_number"] == 2][["series_id", "player_name", stat]].rename(columns={stat: "_stat2"})
    out = m1.merge(m2, on=["series_id", "player_name"], how="inner")
    out["actual_sum"] = out[stat].to_numpy(dtype=float) + out["_stat2"].to_numpy(dtype=float)
    return out


def _sum_p_over(line: float, mus: tuple[float, float], player: str, model, n: int) -> tuple[float, float, float]:
    comps = [Component(mu=float(mus[0]), player=player, team=None, map_index=1), Component(mu=float(mus[1]), player=player, team=None, map_index=2)]
    return sum_over_under_push(line, comps, model.r, model.rho_self, model.rho_team, model.rho_opp, n=n)


def fair_sum_line(mus: tuple[float, float], player: str, model, n: int) -> float:
    """Of floor(mu1 + mu2) - 0.5 and + 0.5, the half-line whose over probability under the sum model is closest to 50%."""
    total = float(mus[0] + mus[1])
    lo, hi = np.floor(total) - 0.5, np.floor(total) + 0.5
    p_lo = _sum_p_over(lo, mus, player, model, n)[0]
    p_hi = _sum_p_over(hi, mus, player, model, n)[0]
    return float(max(0.5, lo if abs(p_lo - 0.5) < abs(p_hi - 0.5) else hi))


def price_fold_two_map(model, pairs: pd.DataFrame, mus: tuple[np.ndarray, np.ndarray], book_mus: tuple[np.ndarray, np.ndarray],
                       shrink: float, threshold: float, n_sim: int = 4000) -> pd.DataFrame:
    rows = []
    y = pairs["actual_sum"].to_numpy(dtype=float)
    players = pairs["player_name"].to_numpy()
    dates = pairs["date"].to_numpy()
    for i in range(len(pairs)):
        line = fair_sum_line((book_mus[0][i], book_mus[1][i]), players[i], model, n_sim)
        share = line / 2.0
        adj = ((1 - shrink) * mus[0][i] + shrink * share, (1 - shrink) * mus[1][i] + shrink * share)
        over, under, push = _sum_p_over(line, adj, players[i], model, n_sim)
        over_c = float(model.calibrate(over))
        under_c = max(0.0, 1.0 - over_c - push)
        lean_over = over_c >= under_c
        prob = over_c if lean_over else under_c
        rows.append({"date": dates[i], "line": line, "mu": mus[0][i] + mus[1][i], "mu_adj": adj[0] + adj[1], "actual": y[i], "prob": prob,
                     "lean_over": lean_over, "pick": prob >= threshold, "hit": (y[i] > line) if lean_over else (y[i] < line)})
    return pd.DataFrame(rows)


def walk_forward_two_map(frame: pd.DataFrame, sport: str, stat: str, months: int = 5, shrink: float = 0.25, threshold: float = 0.60,
                         min_games: int = 3, n_sim: int = 4000, progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward for two-map sums: same folds and setters as `walk_forward`, lines on map 1 + map 2 totals."""
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
        book = fit_booklike(past, stat, cat_levels)
        pairs = two_map_pairs(fold, stat)
        naive_mean = pairs[f"pr_{stat}_mean10"].fillna(pairs[f"p_{stat}_mean10"]).to_numpy(dtype=float)
        ok = ~np.isnan(naive_mean)
        pairs, naive_mean = pairs[ok].reset_index(drop=True), naive_mean[ok]
        if len(pairs) < 50:
            continue
        as_map2 = pairs.copy()
        as_map2["game_number"] = 2
        mus = (model.predict_mu(pairs), model.predict_mu(as_map2))
        book_mus = (predict_booklike(book, pairs, stat, cat_levels), predict_booklike(book, as_map2, stat, cat_levels))
        for book_name, bm in (("naive", (naive_mean, naive_mean)), ("booklike", book_mus)):
            res = price_fold_two_map(model, pairs, mus, bm, shrink, threshold, n_sim)
            res["book"], res["fold"] = book_name, start.strftime("%Y-%m")
            picks.append(res)
            sel = res[res["pick"]]
            n, h = int(len(sel)), int(sel["hit"].sum())
            p, lo, hi = wilson(h, n) if n else (float("nan"),) * 3
            rows.append({"fold": start.strftime("%Y-%m"), "book": book_name, "games": int(len(res)), "picks": n, "hits": h,
                         "hit_rate": p, "ci_low": lo, "ci_high": hi, "parlay4_roi": parlay_roi(p) if n else float("nan"),
                         "book_mae": float(np.mean(np.abs(res["actual"] - res["line"]))), "model_mae": float(np.mean(np.abs(res["actual"] - res["mu"])))})
        if progress:
            progress(f"{sport}/{stat} two-map fold {start:%Y-%m}: trained on {len(past)} rows, priced {len(pairs)} two-map sums")
    summary = pd.DataFrame(rows)
    pooled = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
    return summary, pooled

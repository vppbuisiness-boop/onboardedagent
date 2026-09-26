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
from pathlib import Path

import joblib

import lightgbm as lgb
from scipy import stats
import numpy as np
import pandas as pd

from ..features.build import CATEGORICAL, FEATURE_COLUMNS
from ..grading.roi import parlay_roi, wilson
from .copula import Component, sum_over_under_push
from .distributions import fair_line, over_under_push
from .props import LGB_PARAMS, PropModel, train

CACHE_DIR = Path("data/backtests/cache")
CACHE_VERSION = 1  # bump when props.train or the feature set changes in a way that should invalidate cached fold models


def cached_train(past: pd.DataFrame, sport: str, stat: str) -> PropModel:
    """train() with a joblib cache keyed by sport, stat, feature set and the training window.

    The single-map and two-map walk-forwards fit the same fold models; caching them halves a full sweep."""
    key = f"{sport}_{stat}_v{CACHE_VERSION}_{len(past)}_{pd.Timestamp(past['date'].max()).strftime('%Y%m%d')}_{abs(hash(tuple(FEATURE_COLUMNS))) % 10**8}"
    path = CACHE_DIR / f"{key}.joblib"
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            pass
    model = train(past, sport, stat, valid_frac=0.2)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model

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
        model = cached_train(past, sport, stat)
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


def _two_map_totals(mu1: np.ndarray, mu2: np.ndarray, r: float, rho_self: float, n: int, seed: int = 7, chunk: int = 400) -> np.ndarray:
    """(rows, n) simulated map-1 + map-2 totals under the live Gaussian-copula NB model.

    For one player's two maps the copula correlation is rho_self for every row, so one normal draw is shared and
    the negative-binomial quantile transform is vectorised across rows; this is the same model the live pricer
    simulates per line, without the per-line Python loop."""
    rng = np.random.default_rng(seed)
    rho = float(np.clip(rho_self, -0.95, 0.95))
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    U = np.clip(stats.norm.cdf(rng.standard_normal((n, 2)) @ L.T), 1e-9, 1 - 1e-9)
    out = np.empty((len(mu1), n))
    for a in range(0, len(mu1), chunk):
        m1, m2 = np.maximum(mu1[a:a + chunk], 1e-6)[:, None], np.maximum(mu2[a:a + chunk], 1e-6)[:, None]
        x1 = stats.nbinom.ppf(U[None, :, 0], r, r / (r + m1))
        x2 = stats.nbinom.ppf(U[None, :, 1], r, r / (r + m2))
        out[a:a + chunk] = x1 + x2
    return out


def fair_sum_lines(totals: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Per row: of floor(mean) - 0.5 and + 0.5, the half-line whose simulated over probability is closest to 50%."""
    lo, hi = np.floor(means) - 0.5, np.floor(means) + 0.5
    p_lo = (totals > lo[:, None]).mean(axis=1)
    p_hi = (totals > hi[:, None]).mean(axis=1)
    return np.maximum(0.5, np.where(np.abs(p_lo - 0.5) < np.abs(p_hi - 0.5), lo, hi))


def price_fold_two_map(model, pairs: pd.DataFrame, mus: tuple[np.ndarray, np.ndarray], book_mus: tuple[np.ndarray, np.ndarray],
                       shrink: float, threshold: float, n_sim: int = 4000) -> pd.DataFrame:
    y = pairs["actual_sum"].to_numpy(dtype=float)
    book_tot = _two_map_totals(book_mus[0], book_mus[1], model.r, model.rho_self, n_sim)
    lines = fair_sum_lines(book_tot, book_mus[0] + book_mus[1])
    share = lines / 2.0
    adj1, adj2 = (1 - shrink) * mus[0] + shrink * share, (1 - shrink) * mus[1] + shrink * share
    tot = _two_map_totals(adj1, adj2, model.r, model.rho_self, n_sim, seed=11)
    over = (tot > lines[:, None]).mean(axis=1)
    push = (tot == lines[:, None]).mean(axis=1)
    over_c = np.asarray(model.calibrate(over), dtype=float)
    under_c = np.clip(1.0 - over_c - push, 0.0, None)
    lean_over = over_c >= under_c
    prob = np.where(lean_over, over_c, under_c)
    return pd.DataFrame({"date": pairs["date"].to_numpy(), "line": lines, "mu": mus[0] + mus[1], "mu_adj": adj1 + adj2, "actual": y, "prob": prob,
                         "lean_over": lean_over, "pick": prob >= threshold, "hit": np.where(lean_over, y > lines, y < lines)})


def walk_forward_two_map(frame: pd.DataFrame, sport: str, stat: str, months: int = 5, shrink: float = 0.25, threshold: float = 0.60,
                         min_games: int = 3, n_sim: int = 4000, max_pairs: int = 2500, seed: int = 7, progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward for two-map sums: same folds and setters as `walk_forward`, lines on map 1 + map 2 totals.

    Each sum is priced by simulation (three draws of `n_sim` per pair), so folds are capped at `max_pairs`
    random pairs; CS2 folds hold ~10k pairs and would otherwise take hours."""
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
        model = cached_train(past, sport, stat)
        cat_levels = model.cat_levels
        book = fit_booklike(past, stat, cat_levels)
        pairs = two_map_pairs(fold, stat)
        naive_mean = pairs[f"pr_{stat}_mean10"].fillna(pairs[f"p_{stat}_mean10"]).to_numpy(dtype=float)
        ok = ~np.isnan(naive_mean)
        pairs, naive_mean = pairs[ok].reset_index(drop=True), naive_mean[ok]
        if len(pairs) < 50:
            continue
        if max_pairs and len(pairs) > max_pairs:
            keep = np.sort(np.random.default_rng(seed).choice(len(pairs), size=max_pairs, replace=False))
            pairs, naive_mean = pairs.iloc[keep].reset_index(drop=True), naive_mean[keep]
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

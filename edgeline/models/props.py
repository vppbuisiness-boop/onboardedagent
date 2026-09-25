"""Per-map count model + dispersion + calibration + series correlation, per sport and stat.

Pipeline (train):
 1. Time-ordered split: last `valid_frac` of dates held out.
 2. LightGBM with Poisson objective predicts the per-map mean mu.
 3. NB dispersion r fitted by MLE on held-out (y, mu).
 4. Within-series correlation of standardized residuals -> shared frailty phi.
 5. Calibration: isotonic regression from raw NB tail probabilities to observed
    frequencies on held-out games, at lines placed where books place them
    (around the predicted mean). Reliability table and Brier reported.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from ..config import ARTIFACT_DIR
from ..features.build import CATEGORICAL, FEATURE_COLUMNS
from .distributions import fit_dispersion, frailty_from_corr, nb_var, over_under_push

LGB_PARAMS = {
    "objective": "poisson",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 60,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 7,
}


@dataclass
class PropModel:
    sport: str
    stat: str
    version: str
    booster: lgb.Booster
    feature_columns: list[str]
    categorical: list[str]
    cat_levels: dict[str, list[str]]
    r: float
    phi: float
    iso: IsotonicRegression | None
    metrics: dict = field(default_factory=dict)
    rho_self: float = 0.0
    rho_team: float = 0.0
    rho_opp: float = 0.0

    def _frame(self, rows: pd.DataFrame) -> pd.DataFrame:
        X = rows.reindex(columns=self.feature_columns).copy()
        for c in self.categorical:
            X[c] = pd.Categorical(X[c].astype(str), categories=self.cat_levels[c])
        return X

    def predict_mu(self, rows: pd.DataFrame) -> np.ndarray:
        return np.clip(self.booster.predict(self._frame(rows)), 0.05, None)

    def calibrate(self, p: np.ndarray | float) -> np.ndarray | float:
        if self.iso is None:
            return p
        arr = np.atleast_1d(np.asarray(p, dtype=float))
        out = np.clip(self.iso.predict(arr), 0.001, 0.999)
        return out if np.ndim(p) else float(out[0])

    def save(self, path: Path | None = None) -> Path:
        path = path or ARTIFACT_DIR / f"{self.sport}_{self.stat}.joblib"
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(sport: str, stat: str, path: Path | None = None) -> "PropModel":
        path = path or ARTIFACT_DIR / f"{sport}_{stat}.joblib"
        return joblib.load(path)


def _series_corr(valid: pd.DataFrame, mu: np.ndarray, r: float, stat: str) -> tuple[float, int]:
    """Correlation of standardized residuals between consecutive maps of the same player in one series."""
    v = valid[["player_name", "series_id", "game_number", stat]].copy()
    v["mu"] = mu
    v["z"] = (v[stat] - v["mu"]) / np.sqrt([nb_var(m, r) for m in v["mu"]])
    v = v.dropna(subset=["series_id"])
    v = v[v["series_id"].astype(str) != "<NA>"]
    a = v[v["game_number"] == 1][["player_name", "series_id", "z"]].rename(columns={"z": "z1"})
    b = v[v["game_number"] == 2][["player_name", "series_id", "z"]].rename(columns={"z": "z2"})
    m = a.merge(b, on=["player_name", "series_id"])
    if len(m) < 50:
        return 0.0, len(m)
    return float(np.corrcoef(m["z1"], m["z2"])[0, 1]), len(m)


def _pair_corr(valid: pd.DataFrame, mu: np.ndarray, r: float, stat: str) -> dict:
    """Correlation of standardized residuals between players in the same map: teammates vs opponents."""
    v = valid[["game_id", "player_name", "team", stat]].copy()
    v["z"] = (v[stat].to_numpy(dtype=float) - mu) / np.sqrt([nb_var(m, r) for m in mu])
    v = v.dropna(subset=["z", "team"])
    m = v.merge(v, on="game_id", suffixes=("_a", "_b"))
    m = m[m["player_name_a"] < m["player_name_b"]]
    out = {"rho_team": 0.0, "rho_opp": 0.0, "pairs_team": 0, "pairs_opp": 0}
    for key, mask in (("team", m["team_a"] == m["team_b"]), ("opp", m["team_a"] != m["team_b"])):
        sub = m[mask]
        out[f"pairs_{key}"] = int(len(sub))
        if len(sub) >= 200:
            out[f"rho_{key}"] = float(np.corrcoef(sub["z_a"], sub["z_b"])[0, 1])
    return out


def _naive_line_policy(valid: pd.DataFrame, mu: np.ndarray, r: float, stat: str, model_cal) -> dict:
    """Hit rate when betting against a naive book that sets the line at the player's trailing 10-game mean.

    More honest than lines at the model's own mean: the model only gets credit where it disagrees with
    a simple average, which is closer to how soft esports lines are actually set.
    """
    base = valid[f"p_{stat}_mean10"].to_numpy(dtype=float)
    y = valid[stat].to_numpy(dtype=float)
    ok = ~np.isnan(base)
    lines = np.floor(base[ok]) + 0.5
    raw = np.array([over_under_push(l, m, r)[0] for l, m in zip(lines, mu[ok])])
    cal = model_cal(raw)
    out = {}
    for thr in (0.55, 0.60, 0.65):
        over = cal >= thr
        under = (1 - cal) >= thr
        picks = over | under
        hits = np.where(over, y[ok] > lines, y[ok] < lines)[picks]
        out[f"policy_{int(thr*100)}"] = {"n": int(picks.sum()), "hit_rate": float(hits.mean()) if picks.sum() else None}
    return out


def _calibration_set(valid: pd.DataFrame, mu: np.ndarray, r: float, stat: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic lines near the predicted mean (where books set them): raw P(over) vs outcome."""
    y = valid[stat].to_numpy(dtype=float)
    raws, obs = [], []
    for yi, mi in zip(y, mu):
        base = np.floor(mi)
        for off in (-1.0, 0.0, 1.0):
            line = base + off + 0.5
            if line < 0.5:
                continue
            p_over, _, _ = over_under_push(line, mi, r)
            raws.append(p_over)
            obs.append(1.0 if yi > line else 0.0)
    return np.asarray(raws), np.asarray(obs)


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bucket": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(m.sum()), "pred": float(p[m].mean()), "obs": float(y[m].mean())})
    return rows


def train(frame: pd.DataFrame, sport: str, stat: str, valid_frac: float = 0.2, min_games: int = 3,
          num_rounds: int = 2000, seed: int = 7) -> PropModel:
    df = frame.dropna(subset=[stat]).copy()
    df = df[df["p_games"] >= min_games]
    df = df.sort_values("date")
    dates = df["date"].to_numpy()
    cut = dates[int(len(dates) * (1 - valid_frac))]
    train_df, valid_df = df[df["date"] < cut], df[df["date"] >= cut]
    cat_levels = {c: sorted(df[c].astype(str).unique().tolist()) for c in CATEGORICAL}

    def frame_of(d: pd.DataFrame) -> pd.DataFrame:
        X = d.reindex(columns=FEATURE_COLUMNS + CATEGORICAL).copy()
        for c in CATEGORICAL:
            X[c] = pd.Categorical(X[c].astype(str), categories=cat_levels[c])
        return X

    dtrain = lgb.Dataset(frame_of(train_df), label=train_df[stat].to_numpy(dtype=float), categorical_feature=CATEGORICAL, free_raw_data=False)
    dvalid = lgb.Dataset(frame_of(valid_df), label=valid_df[stat].to_numpy(dtype=float), reference=dtrain, categorical_feature=CATEGORICAL, free_raw_data=False)
    params = {**LGB_PARAMS, "seed": seed}
    booster = lgb.train(params, dtrain, num_boost_round=num_rounds, valid_sets=[dvalid], callbacks=[lgb.early_stopping(100, verbose=False)])

    mu_v = np.clip(booster.predict(frame_of(valid_df), num_iteration=booster.best_iteration), 0.05, None)
    y_v = valid_df[stat].to_numpy(dtype=float)
    r = fit_dispersion(y_v, mu_v)
    corr, n_pairs = _series_corr(valid_df, mu_v, r, stat)
    phi = frailty_from_corr(max(corr, 0.0), float(np.median(mu_v)), r)

    rng = np.random.default_rng(seed)
    raw, obs = _calibration_set(valid_df, mu_v, r, stat, rng)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, obs)
    cal = np.clip(iso.predict(raw), 1e-4, 1 - 1e-4)
    baseline_mu = float(train_df[stat].mean())
    pairs = _pair_corr(valid_df, mu_v, r, stat)
    naive = _naive_line_policy(valid_df, mu_v, r, stat, lambda a: np.clip(iso.predict(a), 1e-4, 1 - 1e-4))
    metrics = {
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "valid_from": str(pd.Timestamp(cut).date()),
        "best_iteration": int(booster.best_iteration),
        "mae_valid": float(np.mean(np.abs(y_v - mu_v))),
        "mae_baseline_player_mean10": float(np.nanmean(np.abs(y_v - valid_df[f"p_{stat}_mean10"].to_numpy(dtype=float)))),
        "mae_baseline_global": float(np.mean(np.abs(y_v - baseline_mu))),
        "dispersion_r": float(r),
        "series_corr": float(corr),
        "series_pairs": int(n_pairs),
        "frailty_phi": float(phi),
        "rho_team": pairs["rho_team"],
        "rho_opp": pairs["rho_opp"],
        "pairs_team": pairs["pairs_team"],
        "pairs_opp": pairs["pairs_opp"],
        "naive_line_policy": naive,
        "brier_raw": float(brier_score_loss(obs, np.clip(raw, 1e-4, 1 - 1e-4))),
        "brier_calibrated": float(brier_score_loss(obs, cal)),
        "logloss_raw": float(log_loss(obs, np.clip(raw, 1e-4, 1 - 1e-4))),
        "logloss_calibrated": float(log_loss(obs, cal)),
        "reliability_raw": reliability_table(raw, obs),
        "reliability_calibrated": reliability_table(cal, obs),
        "hit_rate_at_60_calibrated": _hit_rate_at(cal, obs, 0.60),
        "feature_importance": dict(sorted(zip(booster.feature_name(), booster.feature_importance("gain").round(1).tolist()), key=lambda kv: -kv[1])[:15]),
    }
    version = f"{sport}-{stat}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d%H%M')}"
    return PropModel(sport, stat, version, booster, FEATURE_COLUMNS + CATEGORICAL, CATEGORICAL, cat_levels, float(r), float(phi), iso, metrics,
                     rho_self=float(max(corr, 0.0)), rho_team=float(pairs["rho_team"]), rho_opp=float(pairs["rho_opp"]))


def _hit_rate_at(p: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """Simulated 'bet the side with >= threshold' policy on the calibration set."""
    over = p >= threshold
    under = (1 - p) >= threshold
    picks = over | under
    if picks.sum() == 0:
        return {"n": 0, "hit_rate": None}
    hits = np.where(over, y == 1, y == 0)[picks]
    return {"n": int(picks.sum()), "hit_rate": float(hits.mean())}


def metrics_summary(m: PropModel) -> str:
    mt = m.metrics
    lines = [
        f"{m.sport}/{m.stat} version={m.version}",
        f"  train={mt['n_train']} valid={mt['n_valid']} (from {mt['valid_from']}) best_iter={mt['best_iteration']}",
        f"  MAE model={mt['mae_valid']:.3f} | player mean10={mt['mae_baseline_player_mean10']:.3f} | global={mt['mae_baseline_global']:.3f}",
        f"  NB dispersion r={mt['dispersion_r']:.2f}  rho_self={mt['series_corr']:.3f} (n={mt['series_pairs']})  rho_team={mt.get('rho_team', 0):.3f} (n={mt.get('pairs_team', 0)})  rho_opp={mt.get('rho_opp', 0):.3f} (n={mt.get('pairs_opp', 0)})",
        f"  Brier raw={mt['brier_raw']:.4f} -> calibrated={mt['brier_calibrated']:.4f}; logloss {mt['logloss_raw']:.4f} -> {mt['logloss_calibrated']:.4f}",
        f"  policy >=60% vs lines at model mean: n={mt['hit_rate_at_60_calibrated']['n']} hit_rate={mt['hit_rate_at_60_calibrated']['hit_rate']}",
        "  policy vs naive book (line = trailing 10-game mean): " + ", ".join(f">={k[-2:]}%: n={v['n']} hit={v['hit_rate'] if v['hit_rate'] is None else round(v['hit_rate'], 3)}" for k, v in mt.get("naive_line_policy", {}).items()),
        "  reliability (calibrated): " + ", ".join(f"{r['bucket']}: pred {r['pred']:.2f} obs {r['obs']:.2f} n={r['n']}" for r in mt["reliability_calibrated"]),
        "  top features: " + ", ".join(f"{k}={v:.0f}" for k, v in list(mt["feature_importance"].items())[:8]),
    ]
    return "\n".join(lines)


def save_metrics(m: PropModel) -> Path:
    p = ARTIFACT_DIR / f"{m.sport}_{m.stat}_metrics.json"
    p.write_text(json.dumps(m.metrics, indent=2, default=str))
    return p

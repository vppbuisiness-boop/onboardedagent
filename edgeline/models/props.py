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
    return PropModel(sport, stat, version, booster, FEATURE_COLUMNS + CATEGORICAL, CATEGORICAL, cat_levels, float(r), float(phi), iso, metrics)


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
        f"  NB dispersion r={mt['dispersion_r']:.2f}  series corr={mt['series_corr']:.3f} (pairs={mt['series_pairs']}) phi={mt['frailty_phi']:.4f}",
        f"  Brier raw={mt['brier_raw']:.4f} -> calibrated={mt['brier_calibrated']:.4f}; logloss {mt['logloss_raw']:.4f} -> {mt['logloss_calibrated']:.4f}",
        f"  policy >=60%: n={mt['hit_rate_at_60_calibrated']['n']} hit_rate={mt['hit_rate_at_60_calibrated']['hit_rate']}",
        "  reliability (calibrated): " + ", ".join(f"{r['bucket']}: pred {r['pred']:.2f} obs {r['obs']:.2f} n={r['n']}" for r in mt["reliability_calibrated"]),
        "  top features: " + ", ".join(f"{k}={v:.0f}" for k, v in list(mt["feature_importance"].items())[:8]),
    ]
    return "\n".join(lines)


def save_metrics(m: PropModel) -> Path:
    p = ARTIFACT_DIR / f"{m.sport}_{m.stat}_metrics.json"
    p.write_text(json.dumps(m.metrics, indent=2, default=str))
    return p

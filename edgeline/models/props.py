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
from scipy import stats
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from ..config import ARTIFACT_DIR
from ..features.build import CATEGORICAL, FEATURE_COLUMNS
from .distributions import fair_line, fit_dispersion, frailty_from_corr, nb_var, over_under_push

class LogitCalibrator:
    """Platt-style recalibration on the logit scale: p_cal = sigmoid(a * logit(p_raw) + b).

    Smooth and monotone, so it extrapolates sensibly outside the fitted range (an isotonic
    fit clipped raw probabilities outside its narrow support to 0 or 1).
    """

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a, self.b = float(a), float(b)

    @staticmethod
    def fit(raw: np.ndarray, obs: np.ndarray) -> "LogitCalibrator":
        x = logit(np.clip(raw, 1e-4, 1 - 1e-4)).reshape(-1, 1)
        lr = LogisticRegression(C=1e4, solver="lbfgs").fit(x, obs.astype(int))
        return LogitCalibrator(lr.coef_[0][0], lr.intercept_[0])

    def predict(self, raw) -> np.ndarray:
        x = logit(np.clip(np.asarray(raw, dtype=float), 1e-4, 1 - 1e-4))
        return expit(self.a * x + self.b)


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
    calibrator: LogitCalibrator | None
    metrics: dict = field(default_factory=dict)
    rho_self: float = 0.0
    rho_team: float = 0.0
    rho_opp: float = 0.0
    mean_bias: float = 1.0  # validation-split ratio mean(actual) / mean(predicted); corrects the GBM's low mean

    def _frame(self, rows: pd.DataFrame) -> pd.DataFrame:
        X = rows.reindex(columns=self.feature_columns).copy()
        for c in self.categorical:
            X[c] = pd.Categorical(X[c].astype(str), categories=self.cat_levels[c])
        return X

    def predict_mu(self, rows: pd.DataFrame) -> np.ndarray:
        return np.clip(self.booster.predict(self._frame(rows)) * getattr(self, "mean_bias", 1.0), 0.05, None)

    def calibrate(self, p: np.ndarray | float) -> np.ndarray | float:
        if self.calibrator is None:
            return p
        arr = np.atleast_1d(np.asarray(p, dtype=float))
        out = np.clip(self.calibrator.predict(arr), 0.001, 0.999)
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
    """Hit rate when betting against a naive book that sets the line at the median-fair half nearest the player's trailing 10-game mean.

    More honest than lines at the model's own mean: the model only gets credit where it disagrees with
    a simple average, which is closer to how soft esports lines are actually set.
    """
    # per (player, role) trailing mean where available (role = game mode in COD), else per player
    base = valid[f"pr_{stat}_mean10"].to_numpy(dtype=float) if f"pr_{stat}_mean10" in valid.columns else valid[f"p_{stat}_mean10"].to_numpy(dtype=float)
    fallback = valid[f"p_{stat}_mean10"].to_numpy(dtype=float)
    base = np.where(np.isnan(base), fallback, base)
    y = valid[stat].to_numpy(dtype=float)
    ok = ~np.isnan(base)
    lines = fair_line(base[ok], r)  # median-fair half-line, not mean rounded up (see distributions.fair_line)
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


CAL_OFFSETS = (-4, -3, -2, -1, 0, 1, 2, 3, 4)


def _nb_sf(lines: np.ndarray, mu: np.ndarray, r: np.ndarray | float) -> np.ndarray:
    """P(X > line) for half-integer lines under NB(mu, r), vectorized."""
    r = np.asarray(r, dtype=float)
    return stats.nbinom.sf(np.floor(lines), r, r / (r + np.clip(mu, 1e-6, None)))


def _calibration_set(valid: pd.DataFrame, mu: np.ndarray, r: float, stat: str, rho_self: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Raw tail probability vs outcome at lines spread widely around the mean, for single maps and
    two-map sums (moment-matched NB), so the calibrator covers the range the board asks for."""
    y = valid[stat].to_numpy(dtype=float)
    raws, obs = [], []
    base = np.floor(mu)
    for k in CAL_OFFSETS:
        lines = base + k + 0.5
        ok = lines >= 0.5
        raws.append(_nb_sf(lines[ok], mu[ok], r))
        obs.append((y[ok] > lines[ok]).astype(float))
    v = valid[["player_name", "series_id", "game_number"]].copy()
    v["mu"], v["y"], v["i"] = mu, y, np.arange(len(v))
    v = v[v["series_id"].notna()]
    nxt = v.copy()
    nxt["game_number"] = nxt["game_number"] - 1
    pair = v.merge(nxt, on=["player_name", "series_id", "game_number"], suffixes=("_a", "_b"))
    if len(pair):
        mu_a, mu_b = pair["mu_a"].to_numpy(), pair["mu_b"].to_numpy()
        var_a, var_b = mu_a + mu_a**2 / r, mu_b + mu_b**2 / r
        m = mu_a + mu_b
        var = var_a + var_b + 2 * rho_self * np.sqrt(var_a * var_b)
        r_eff = np.where(var > m + 1e-6, m**2 / np.clip(var - m, 1e-6, None), 1e6)
        ysum = pair["y_a"].to_numpy() + pair["y_b"].to_numpy()
        for k in CAL_OFFSETS:
            lines = np.floor(m) + k + 0.5
            ok = lines >= 0.5
            raws.append(_nb_sf(lines[ok], m[ok], r_eff[ok]))
            obs.append((ysum[ok] > lines[ok]).astype(float))
    return np.concatenate(raws), np.concatenate(obs)


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


def load_tuned_params(sport: str) -> dict:
    p = ARTIFACT_DIR / f"{sport}_params.json"
    return json.loads(p.read_text()) if p.exists() else {}


RECENCY_HALFLIFE_DAYS: float | None = None  # set via train(recency_halflife=...) once validated


def train(frame: pd.DataFrame, sport: str, stat: str, valid_frac: float = 0.2, min_games: int = 3,
          num_rounds: int = 2000, seed: int = 7, params: dict | None = None, recency_halflife: float | None = RECENCY_HALFLIFE_DAYS) -> PropModel:
    df = frame.dropna(subset=[stat]).copy()
    df = df[(df[stat] >= 0) & (df["p_games"] >= min_games)]  # negative counts are source glitches
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

    weight = None
    if recency_halflife:
        age_days = (pd.Timestamp(cut) - train_df["date"]).dt.total_seconds().to_numpy() / 86400.0
        weight = 0.5 ** (np.clip(age_days, 0, None) / float(recency_halflife))
    dtrain = lgb.Dataset(frame_of(train_df), label=train_df[stat].to_numpy(dtype=float), weight=weight, categorical_feature=CATEGORICAL, free_raw_data=False)
    dvalid = lgb.Dataset(frame_of(valid_df), label=valid_df[stat].to_numpy(dtype=float), reference=dtrain, categorical_feature=CATEGORICAL, free_raw_data=False)
    params = {**LGB_PARAMS, **(params or load_tuned_params(sport).get(stat, {})), "seed": seed}
    booster = lgb.train(params, dtrain, num_boost_round=num_rounds, valid_sets=[dvalid], callbacks=[lgb.early_stopping(100, verbose=False)])

    mu_raw = np.clip(booster.predict(frame_of(valid_df), num_iteration=booster.best_iteration), 0.05, None)
    y_v = valid_df[stat].to_numpy(dtype=float)
    # The Poisson GBM's means run 1% to 4% low out of sample (early stopping, Jensen's gap on the log link,
    # and slow upward drift in kills per map); with a right-skewed NB that bias alone makes the model lean
    # UNDER on lines set at the true median. Rescale by the held-out ratio before fitting r, the
    # calibrator and the correlations, so every downstream quantity sees the corrected mean.
    # Level-sensitive quantities (mean bias, dispersion, the probability calibrator) are estimated on the most
    # recent window of the held-out split: the level of kills drifts within a season, and a calibrator fit
    # across the whole split learns the average of the drift, which biased every probability toward UNDER
    # in the months that followed (walk-forward: predicted P(over) 0.52 -> observed 0.60 in Dota).
    recent = _recent_mask(valid_df["date"].to_numpy())
    mean_bias = _mean_bias(valid_df["date"].to_numpy(), y_v, mu_raw)
    mu_v = np.clip(mu_raw * mean_bias, 0.05, None)
    r = fit_dispersion(y_v[recent], mu_v[recent])
    corr, n_pairs = _series_corr(valid_df, mu_v, r, stat)
    phi = frailty_from_corr(max(corr, 0.0), float(np.median(mu_v)), r)

    raw, obs = _calibration_set(valid_df[recent], mu_v[recent], r, stat, rho_self=max(corr, 0.0))
    calibrator = LogitCalibrator.fit(raw, obs)
    cal = np.clip(calibrator.predict(raw), 1e-4, 1 - 1e-4)
    baseline_mu = float(train_df[stat].mean())
    pairs = _pair_corr(valid_df, mu_v, r, stat)
    naive = _naive_line_policy(valid_df, mu_v, r, stat, lambda a: np.clip(calibrator.predict(a), 1e-4, 1 - 1e-4))
    metrics = {
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "valid_from": str(pd.Timestamp(cut).date()),
        "best_iteration": int(booster.best_iteration),
        "mae_valid": float(np.mean(np.abs(y_v - mu_v))),
        "mean_bias": mean_bias,
        "mae_valid_uncorrected": float(np.mean(np.abs(y_v - mu_raw))),
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
        "calibrator": {"a": calibrator.a, "b": calibrator.b, "n_samples": int(len(raw)), "window_days": BIAS_WINDOW_DAYS, "window_rows": int(recent.sum())},
        "params": {k: params[k] for k in ("num_leaves", "learning_rate", "min_data_in_leaf", "feature_fraction", "lambda_l2")},
        "recency_halflife_days": recency_halflife,
        "valid_poisson_nll": float(np.mean(mu_v - y_v * np.log(mu_v))),
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
    return PropModel(sport, stat, version, booster, FEATURE_COLUMNS + CATEGORICAL, CATEGORICAL, cat_levels, float(r), float(phi), calibrator, metrics,
                     rho_self=float(max(corr, 0.0)), rho_team=float(pairs["rho_team"]), rho_opp=float(pairs["rho_opp"]), mean_bias=mean_bias)


BIAS_WINDOW_DAYS = 90
BIAS_MIN_ROWS = 500


def _recent_mask(dates: np.ndarray, window_days: int = BIAS_WINDOW_DAYS, min_rows: int = BIAS_MIN_ROWS) -> np.ndarray:
    """Rows within `window_days` of the latest date; every row when that window holds fewer than `min_rows`."""
    if len(dates) == 0:
        return np.zeros(0, dtype=bool)
    ts = pd.to_datetime(pd.Series(dates), utc=True)
    recent = (ts >= ts.max() - pd.Timedelta(days=window_days)).to_numpy()
    return recent if recent.sum() >= min_rows else np.ones(len(dates), dtype=bool)


def _mean_bias(dates: np.ndarray, y: np.ndarray, mu: np.ndarray, window_days: int = BIAS_WINDOW_DAYS, min_rows: int = BIAS_MIN_ROWS) -> float:
    """mean(actual) / mean(predicted) over the most recent `window_days` of the held-out split, clipped to 0.9-1.1.

    The level of kills drifts within a season (Dota fell from 5.9 to 5.0 per game over the 2025-26 winter and
    climbed back to 5.7 by September), so a ratio over the whole split can point the wrong way for the months
    that follow; the recent window tracks the current level. Falls back to the whole split when it is thin."""
    if len(y) == 0:
        return 1.0
    recent = _recent_mask(dates, window_days, min_rows)
    return float(np.clip(np.mean(y[recent]) / np.mean(mu[recent]), 0.9, 1.1))


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
        f"  Brier raw={mt['brier_raw']:.4f} -> calibrated={mt['brier_calibrated']:.4f}; logloss {mt['logloss_raw']:.4f} -> {mt['logloss_calibrated']:.4f}; calibrator a={mt['calibrator']['a']:.3f} b={mt['calibrator']['b']:+.3f} (n={mt['calibrator']['n_samples']})",
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


TUNE_GRID = [
    {"num_leaves": 15, "learning_rate": 0.03, "min_data_in_leaf": 60},
    {"num_leaves": 31, "learning_rate": 0.03, "min_data_in_leaf": 60},
    {"num_leaves": 63, "learning_rate": 0.03, "min_data_in_leaf": 60},
    {"num_leaves": 31, "learning_rate": 0.03, "min_data_in_leaf": 200},
    {"num_leaves": 31, "learning_rate": 0.01, "min_data_in_leaf": 60},
    {"num_leaves": 127, "learning_rate": 0.02, "min_data_in_leaf": 100, "feature_fraction": 0.6},
]


def tune(frame: pd.DataFrame, sport: str, stat: str, grid: list[dict] | None = None, valid_frac: float = 0.2) -> tuple[dict, list[dict]]:
    """Try a small grid, keep the config with the lowest validation Poisson deviance; persist per sport/stat."""
    results = []
    best, best_nll = None, float("inf")
    for cfg in grid or TUNE_GRID:
        m = train(frame, sport, stat, valid_frac=valid_frac, params=cfg)
        nll = m.metrics["valid_poisson_nll"]
        pol = m.metrics["naive_line_policy"].get("policy_60", {})
        results.append({**cfg, "valid_poisson_nll": nll, "mae": m.metrics["mae_valid"], "policy60_hit": pol.get("hit_rate"), "policy60_n": pol.get("n"), "best_iteration": m.metrics["best_iteration"]})
        if nll < best_nll:
            best, best_nll = cfg, nll
    p = ARTIFACT_DIR / f"{sport}_params.json"
    current = json.loads(p.read_text()) if p.exists() else {}
    current[stat] = best
    p.write_text(json.dumps(current, indent=2))
    return best, results

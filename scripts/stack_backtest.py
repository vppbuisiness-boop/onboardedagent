"""Correlated same-map stacks vs fixed parlay payouts, walk-forward.

For each monthly fold: cached fold model + a fresh book-like setter; every map in the fold gets a fair line per
player; per map and direction (all UNDER or all OVER) the k legs with the highest calibrated probability form a
stack. Reports realized stack hit rate vs the independent product of leg probabilities vs the copula's joint
probability, and the implied POWER ROI per stack size.

Usage: python scripts/stack_backtest.py cs2 kills [sizes=3,4,5] [min_leg_prob=0.55]
"""
from __future__ import annotations

import sqlite3
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from edgeline.features.build import build_training_frame  # noqa: E402
from edgeline.grading.roi import wilson  # noqa: E402
from edgeline.ev.payouts import ladder  # noqa: E402
from edgeline.models.backtest import cached_train, fit_booklike, predict_booklike  # noqa: E402
from edgeline.models.copula import Component, joint_hit_probability  # noqa: E402
from edgeline.models.distributions import fair_line, over_under_push  # noqa: E402

warnings.filterwarnings("ignore")
sport, stat = sys.argv[1], sys.argv[2]
sizes = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "3,4,5").split(",")]
min_leg = float(sys.argv[4]) if len(sys.argv) > 4 else 0.55
SHRINK, MONTHS, NSIM = 0.25, 5, 3000

conn = sqlite3.connect("data/edgeline.db", timeout=60)
pg = pd.read_sql_query("SELECT * FROM player_games WHERE sport=?", conn, params=(sport,))
frame = build_training_frame(pg)
df = frame.dropna(subset=[stat]); df = df[(df[stat] >= 0) & (df["p_games"] >= 3)].sort_values("date")
last = df["date"].max()
starts = [(last.normalize().replace(day=1) - pd.DateOffset(months=k)) for k in range(MONTHS - 1, -1, -1)]
rows = []
for i, start in enumerate(starts):
    end = starts[i + 1] if i + 1 < len(starts) else last + pd.Timedelta(days=1)
    past, fold = df[df["date"] < start], df[(df["date"] >= start) & (df["date"] < end)]
    if len(past) < 2000 or len(fold) < 200:
        continue
    model = cached_train(past, sport, stat)
    book = fit_booklike(past, stat, model.cat_levels)
    fold = fold.dropna(subset=["team", "game_id"]).copy()
    fold["mu"] = model.predict_mu(fold)
    fold["line"] = fair_line(predict_booklike(book, fold, stat, model.cat_levels), model.r)
    fold["mu_adj"] = (1 - SHRINK) * fold["mu"] + SHRINK * fold["line"]
    p_raw = np.array([over_under_push(l, m, model.r)[0] for l, m in zip(fold["line"], fold["mu_adj"])])
    fold["p_over"] = np.asarray(model.calibrate(p_raw)); fold["p_under"] = 1 - fold["p_over"]
    fold["y"] = fold[stat].to_numpy(dtype=float)
    n_maps = 0
    for gid, g in fold.groupby("game_id"):
        if len(g) < max(sizes):
            continue
        n_maps += 1
        for side in ("UNDER", "OVER"):
            pcol = "p_under" if side == "UNDER" else "p_over"
            cand = g.sort_values(pcol, ascending=False)
            for k in sizes:
                legs = cand.head(k)
                if len(legs) < k or legs[pcol].min() < min_leg:
                    continue
                comps = [([Component(mu=float(r.mu_adj), player=r.player_name, team=r.team, map_index=1)], float(r.line), side) for r in legs.itertuples()]
                joint, _ = joint_hit_probability(comps, model.r, model.rho_self, model.rho_team, model.rho_opp, n=NSIM)
                indep = float(np.prod(legs[pcol].to_numpy()))
                hit = bool(np.all((legs["y"] < legs["line"]) if side == "UNDER" else (legs["y"] > legs["line"])))
                same_team = legs["team"].nunique() == 1
                rows.append({"fold": start.strftime("%Y-%m"), "game_id": gid, "side": side, "k": k, "indep": indep, "copula": joint, "hit": hit,
                             "same_team": same_team, "min_leg": float(legs[pcol].min()), "mean_leg": float(legs[pcol].mean())})
    print(f"fold {start:%Y-%m}: {n_maps} maps, {len(rows)} stacks so far", flush=True)
out = pd.DataFrame(rows)
out.to_csv(f"data/backtests/stacks_{sport}_{stat}.csv", index=False)
print("\nstack results (all folds), POWER ladders: 3-pick %.1fx 4-pick %.1fx 5-pick %.1fx" % tuple(ladder("prizepicks", "POWER", k)[-1] + 1 for k in (3, 4, 5)))
print(f"{'k':>2s} {'side':6s} {'n':>5s} {'realized':>9s} {'ci':>15s} {'indep':>7s} {'copula':>7s} {'ROI@ladder':>11s} {'ROI if indep':>13s}")
for (k, side), g in out.groupby(["k", "side"]):
    n, h = len(g), int(g["hit"].sum()); p, lo, hi = wilson(h, n)
    pay = ladder("prizepicks", "POWER", k)[-1] + 1
    print(f"{k:2d} {side:6s} {n:5d} {p:9.1%} {lo:6.1%}-{hi:6.1%} {g['indep'].mean():7.1%} {g['copula'].mean():7.1%} {p * pay - 1:+11.0%} {g['indep'].mean() * pay - 1:+13.0%}")
print("\nsame-team stacks only:")
for (k, side), g in out[out["same_team"]].groupby(["k", "side"]):
    n, h = len(g), int(g["hit"].sum()); p, lo, hi = wilson(h, n); pay = ladder("prizepicks", "POWER", k)[-1] + 1
    print(f"{k:2d} {side:6s} {n:5d} {p:9.1%} {lo:6.1%}-{hi:6.1%} {g['indep'].mean():7.1%} {g['copula'].mean():7.1%} {p * pay - 1:+11.0%}")
print("STACKS_DONE")

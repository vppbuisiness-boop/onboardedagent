"""Correlated stacks: price pairs of legs from the same game jointly.

Fixed-payout apps treat legs as independent and only shade payouts by a blunt
rule. We compute the true joint hit probability of two legs with the copula
(teammate / opponent / same-player correlations) and report:
  joint      P(both hit) with dependence
  indep      p1 * p2
  lift       joint / indep - 1
  breakeven  the 2-pick payout multiplier at which the stack is zero EV (1 / joint)
  ev_std     EV of the stack as a 2-pick POWER at the book's standard multiplier
Compare `breakeven` with the shaded multiplier the app shows when you add the
pair to a slip: if the app pays more than breakeven, the stack is +EV.

Only pairs priced by the same stat model are combined (cross-stat pairs need
kills-vs-deaths correlations that are not estimated yet).
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd

from ..ev.payouts import ladder
from .copula import joint_hit_probability
from .predict import BoardPricer, PricedLine


def pair_metrics(a: PricedLine, b: PricedLine, model, std_multiplier: float) -> dict:
    joint_raw, (pa_raw, pb_raw) = joint_hit_probability(
        [(a.components, a.line, a.lean), (b.components, b.line, b.lean)], model.r, model.rho_self, model.rho_team, model.rho_opp
    )
    # carry the raw dependence lift onto the calibrated marginals
    indep_raw = max(pa_raw * pb_raw, 1e-9)
    lift = joint_raw / indep_raw
    joint = min(a.prob * b.prob * lift, min(a.prob, b.prob))
    indep = a.prob * b.prob
    same_team = any(ca.team == cb.team and ca.team is not None for ca in a.components for cb in b.components)
    return {
        "joint": float(joint),
        "indep": float(indep),
        "lift": float(joint / indep - 1.0) if indep else float("nan"),
        "breakeven_mult": float(1.0 / joint) if joint > 0 else float("inf"),
        "ev_std": float(joint * std_multiplier - 1.0),
        "relation": "teammates" if same_team else "opponents",
        "direction": "same" if a.lean == b.lean else "opposite",
    }


def find_stacks(pricer: BoardPricer, min_prob: float = 0.55, bettable_only: bool = False, max_pairs_per_game: int = 20) -> pd.DataFrame:
    if not pricer.priced:
        pricer.price(store=False)
    std_mult = ladder(pricer.book, "POWER", 2)[-1] + 1.0
    legs = [p for p in pricer.priced.values() if p.prob >= min_prob and (p.bettable or not bettable_only)]
    by_game: dict[str, list[PricedLine]] = {}
    for p in legs:
        if p.game_id:
            by_game.setdefault(p.game_id, []).append(p)
    meta = pricer.lines.set_index("projection_id")
    rows = []
    for gid, ps in by_game.items():
        pairs = [(a, b) for a, b in combinations(ps, 2) if a.stat == b.stat and a.components[0].player != b.components[0].player]
        for a, b in pairs[:max_pairs_per_game]:
            m = pair_metrics(a, b, pricer.models[a.stat], std_mult)
            ra, rb = meta.loc[a.projection_id], meta.loc[b.projection_id]
            rows.append({
                "game": f"{ra['team']} vs {ra['opponent']}", "leg_a": f"{ra['player_name']} {ra['stat_type']} {a.lean} {a.line:g} ({a.prob:.0%})",
                "leg_b": f"{rb['player_name']} {rb['stat_type']} {b.lean} {b.line:g} ({b.prob:.0%})", **m,
                "ids": f"{a.projection_id},{b.projection_id}",
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["ev_std", "lift"], ascending=False).reset_index(drop=True)

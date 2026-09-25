"""Joint simulation of several negative-binomial prop components with a Gaussian copula.

Components are (player, team, map) cells. Their dependence is described by three
sport-level correlations estimated from standardized model residuals:
  rho_self  same player, different maps of one series
  rho_team  teammates in the same map
  rho_opp   opponents in the same map
Cross-player, cross-map pairs are attenuated (rho_pair * rho_self).

This is what prices multi-map sums, combo props (A + B kills) and correlated
stacks (two legs in the same game) consistently. Marginals stay exactly NB.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Component:
    mu: float
    player: str
    team: str | None
    map_index: int


def build_corr(components: list[Component], rho_self: float, rho_team: float, rho_opp: float) -> np.ndarray:
    k = len(components)
    C = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = components[i], components[j]
            if a.player == b.player:
                rho = rho_self if a.map_index != b.map_index else 1.0
            else:
                pair = rho_team if (a.team is not None and a.team == b.team) else rho_opp
                rho = pair if a.map_index == b.map_index else pair * rho_self
            C[i, j] = C[j, i] = float(np.clip(rho, -0.95, 0.95))
    return nearest_psd(C)


def nearest_psd(C: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    w, v = np.linalg.eigh((C + C.T) / 2)
    w = np.clip(w, eps, None)
    M = (v * w) @ v.T
    d = np.sqrt(np.diag(M))
    M = M / np.outer(d, d)
    np.fill_diagonal(M, 1.0)
    return M


def simulate(components: list[Component], r: float, corr: np.ndarray, n: int = 20000, rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw an (n, k) matrix of NB counts with the given copula correlation."""
    rng = rng or np.random.default_rng()
    k = len(components)
    if k == 0:
        return np.zeros((n, 0))
    L = np.linalg.cholesky(corr)
    Z = rng.standard_normal((n, k)) @ L.T
    U = np.clip(stats.norm.cdf(Z), 1e-9, 1 - 1e-9)
    X = np.empty((n, k))
    for i, c in enumerate(components):
        p = r / (r + max(c.mu, 1e-6))
        X[:, i] = stats.nbinom.ppf(U[:, i], r, p)
    return X


def sum_over_under_push(line: float, components: list[Component], r: float, rho_self: float, rho_team: float, rho_opp: float,
                        n: int = 20000, seed: int | None = 7) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    corr = build_corr(components, rho_self, rho_team, rho_opp)
    total = simulate(components, r, corr, n, rng).sum(axis=1)
    over = float(np.mean(total > line))
    under = float(np.mean(total < line))
    return over, under, max(0.0, 1.0 - over - under)


def joint_hit_probability(legs: list[tuple[list[Component], float, str]], r: float, rho_self: float, rho_team: float, rho_opp: float,
                          n: int = 30000, seed: int | None = 7) -> tuple[float, list[float]]:
    """P(every leg hits) and each leg's marginal, for legs = [(components, line, 'OVER'|'UNDER'), ...] sharing one stat model."""
    rng = np.random.default_rng(seed)
    flat: list[Component] = []
    spans: list[tuple[int, int]] = []
    for comps, _, _ in legs:
        spans.append((len(flat), len(flat) + len(comps)))
        flat.extend(comps)
    corr = build_corr(flat, rho_self, rho_team, rho_opp)
    X = simulate(flat, r, corr, n, rng)
    hits = np.ones(n, dtype=bool)
    marginals = []
    for (a, b), (_, line, side) in zip(spans, legs):
        total = X[:, a:b].sum(axis=1)
        h = total > line if side == "OVER" else total < line
        marginals.append(float(h.mean()))
        hits &= h
    return float(hits.mean()), marginals

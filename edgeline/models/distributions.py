"""Count distributions for prop pricing: negative binomial per map and correlated
multi-map sums via a shared gamma frailty.

Marginal per-map model: X ~ NB(mean mu, dispersion r), Var = mu + mu^2 / r.

Multi-map model: G ~ Gamma(shape=1/phi, scale=phi) (mean 1, var phi) shared across
the maps of a series (and across players in the same game), and
X_i | G ~ NB(mean mu_i * G, dispersion r_c). Choosing
    r_c = (1 + phi) / (1/r - phi)
keeps the marginal variance of each map equal to the single-map NB fit, while
Corr(X_i, X_j) = mu_i mu_j phi / sqrt(Var_i Var_j) > 0.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar


def nb_params(mu: float, r: float) -> tuple[float, float]:
    """scipy nbinom (n, p) parameterization from mean/dispersion."""
    p = r / (r + mu)
    return r, p


def nb_cdf(k: float, mu: float, r: float) -> float:
    n, p = nb_params(mu, r)
    return float(stats.nbinom.cdf(np.floor(k), n, p))


def nb_pmf(k: int, mu: float, r: float) -> float:
    n, p = nb_params(mu, r)
    return float(stats.nbinom.pmf(k, n, p))


def over_under_push(line: float, mu: float, r: float) -> tuple[float, float, float]:
    """P(X > line), P(X < line), P(X == line) for a single-map NB."""
    if abs(line - round(line)) < 1e-9:
        k = int(round(line))
        push = nb_pmf(k, mu, r)
        under = nb_cdf(k - 1, mu, r)
        over = 1.0 - under - push
        return max(over, 0.0), max(under, 0.0), max(push, 0.0)
    under = nb_cdf(np.floor(line), mu, r)
    return 1.0 - under, under, 0.0


def fit_dispersion(y: np.ndarray, mu: np.ndarray, lo: float = 0.2, hi: float = 200.0) -> float:
    """MLE of the NB dispersion r given observed counts and predicted means."""
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, None)

    def nll(log_r: float) -> float:
        r = float(np.exp(log_r))
        n, p = r, r / (r + mu)
        return -float(np.sum(stats.nbinom.logpmf(y, n, p)))

    res = minimize_scalar(nll, bounds=(np.log(lo), np.log(hi)), method="bounded")
    return float(np.exp(res.x))


def nb_var(mu: float, r: float) -> float:
    return mu + mu * mu / r


def frailty_from_corr(target_corr: float, mu: float, r: float) -> float:
    """Solve for phi so two maps with mean mu have the target correlation (bisection)."""
    if target_corr <= 0:
        return 0.0
    v = nb_var(mu, r)
    lo, hi = 0.0, min(0.999 / r, 5.0)  # need 1/r > phi for a valid conditional dispersion
    for _ in range(80):
        mid = (lo + hi) / 2
        corr = mu * mu * mid / v
        if corr < target_corr:
            lo = mid
        else:
            hi = mid
    return lo


def conditional_dispersion(r: float, phi: float) -> float:
    inv = 1.0 / r - phi
    if inv <= 1e-6:
        return 1e6  # essentially Poisson given G
    return (1.0 + phi) / inv


def simulate_sum(mus: list[float], r: float, phi: float, n: int = 20000, rng: np.random.Generator | None = None,
                 groups: list[int] | None = None) -> np.ndarray:
    """Monte Carlo draws of the sum over maps (and players) with shared frailty.

    mus: per-component means. groups: optional group id per component; components in the
    same group share one G draw (e.g. same series or same game). Default: all share.
    """
    rng = rng or np.random.default_rng()
    mus = np.asarray(mus, dtype=float)
    k = len(mus)
    groups = np.asarray(groups if groups is not None else [0] * k)
    total = np.zeros(n)
    if phi <= 0:
        for mu in mus:
            nn, p = nb_params(mu, r)
            total += rng.negative_binomial(nn, p, size=n)
        return total
    r_c = conditional_dispersion(r, phi)
    g_draws = {}
    for gid in np.unique(groups):
        g_draws[gid] = rng.gamma(1.0 / phi, phi, size=n)
    for mu, gid in zip(mus, groups):
        lam = mu * g_draws[gid]
        if r_c >= 1e5:
            total += rng.poisson(lam)
        else:
            nn = r_c
            p = nn / (nn + lam)
            total += rng.negative_binomial(nn, p)
    return total


def sum_over_under_push(line: float, mus: list[float], r: float, phi: float, n: int = 20000, seed: int | None = 7,
                        groups: list[int] | None = None) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = simulate_sum(mus, r, phi, n=n, rng=rng, groups=groups)
    over = float(np.mean(draws > line))
    under = float(np.mean(draws < line))
    push = max(0.0, 1.0 - over - under)
    return over, under, push

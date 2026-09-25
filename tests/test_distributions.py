import numpy as np
import pytest
from scipy import stats

from edgeline.models.distributions import (
    conditional_dispersion,
    fit_dispersion,
    frailty_from_corr,
    nb_var,
    over_under_push,
    simulate_sum,
    sum_over_under_push,
)


def test_over_under_push_half_line():
    over, under, push = over_under_push(4.5, mu=4.0, r=10.0)
    assert push == 0.0
    assert over + under == pytest.approx(1.0)
    assert over == pytest.approx(1 - stats.nbinom.cdf(4, 10.0, 10.0 / 14.0))


def test_over_under_push_integer_line():
    over, under, push = over_under_push(4.0, mu=4.0, r=10.0)
    assert over + under + push == pytest.approx(1.0)
    assert push > 0.1


def test_fit_dispersion_recovers_truth():
    rng = np.random.default_rng(1)
    mu = rng.uniform(2, 8, size=20000)
    r_true = 6.0
    y = rng.negative_binomial(r_true, r_true / (r_true + mu))
    assert fit_dispersion(y, mu) == pytest.approx(r_true, rel=0.1)


def test_frailty_preserves_marginal_variance_and_adds_correlation():
    mu, r = 5.0, 6.0
    phi = frailty_from_corr(0.25, mu, r)
    rng = np.random.default_rng(3)
    n = 200000
    g = rng.gamma(1 / phi, phi, size=n)
    r_c = conditional_dispersion(r, phi)
    x1 = rng.negative_binomial(r_c, r_c / (r_c + mu * g))
    x2 = rng.negative_binomial(r_c, r_c / (r_c + mu * g))
    assert x1.var() == pytest.approx(nb_var(mu, r), rel=0.05)
    assert np.corrcoef(x1, x2)[0, 1] == pytest.approx(0.25, abs=0.03)


def test_simulate_sum_mean():
    draws = simulate_sum([4.0, 5.0, 6.0], r=8.0, phi=0.02, n=100000, rng=np.random.default_rng(0))
    assert draws.mean() == pytest.approx(15.0, rel=0.02)
    over, under, push = sum_over_under_push(14.5, [4.0, 5.0, 6.0], 8.0, 0.02)
    assert over + under + push == pytest.approx(1.0)
    assert 0.4 < over < 0.7

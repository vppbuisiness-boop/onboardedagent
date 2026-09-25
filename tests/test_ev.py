import math

import numpy as np
import pytest

from edgeline.ev.math import (
    breakeven_hit_rate,
    kelly_fraction,
    leg_ev,
    poisson_binomial_pmf,
    slip_ev,
    slip_ev_common_p,
    slip_hit_prob,
)
from edgeline.ev.payouts import ladder, leg_decimal_odds


def test_four_pick_power_at_60_matches_larry_headline():
    # LCSLarry's "~29% ROI on 4-leg parlays" is 0.6^4 * 10 - 1
    assert slip_ev([0.6] * 4, "prizepicks", "POWER") == pytest.approx(0.6**4 * 10 - 1, abs=1e-12)
    assert slip_ev([0.6] * 4, "prizepicks", "POWER") == pytest.approx(0.296, abs=1e-6)


def test_common_p_equals_poisson_binomial():
    net = ladder("prizepicks", "FLEX", 5)
    for p in (0.5, 0.58, 0.63):
        assert slip_ev_common_p(p, net) == pytest.approx(slip_ev([p] * 5, "prizepicks", "FLEX"), abs=1e-12)


def test_poisson_binomial_pmf_sums_to_one_and_matches_binomial():
    pmf = poisson_binomial_pmf([0.6, 0.6, 0.6])
    assert pmf.sum() == pytest.approx(1.0)
    assert pmf[3] == pytest.approx(0.6**3)
    assert pmf[0] == pytest.approx(0.4**3)
    mixed = poisson_binomial_pmf([0.9, 0.1])
    assert mixed[2] == pytest.approx(0.09)
    assert mixed[1] == pytest.approx(0.9 * 0.9 + 0.1 * 0.1)


def test_leg_odds_from_reference_parlay():
    assert leg_decimal_odds("prizepicks") == pytest.approx(10 ** 0.25, abs=1e-9)  # 1.778
    assert leg_decimal_odds("underdog") == pytest.approx(10 ** 0.25, abs=1e-9)


def test_breakeven_rates():
    assert breakeven_hit_rate(ladder("prizepicks", "POWER", 4)) == pytest.approx(0.1**0.25, abs=1e-4)  # 56.2%
    assert breakeven_hit_rate(ladder("prizepicks", "FLEX", 6)) == pytest.approx(0.542, abs=0.002)


def test_leg_ev_sign():
    assert leg_ev(0.6, 1.778) > 0
    assert leg_ev(0.5, 1.778) < 0
    assert leg_ev(1 / 1.778, 1.778) == pytest.approx(0.0, abs=1e-12)


def test_slip_hit_prob_flex_counts_partial_payouts():
    # 5-flex pays (net > 0) on 5/5 and 4/5 only (3/5 returns 0.4x => net -0.6)
    p = 0.6
    hp = slip_hit_prob([p] * 5, "prizepicks", "FLEX")
    expected = p**5 + 5 * p**4 * (1 - p)
    assert hp == pytest.approx(expected)


def test_kelly():
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)
    assert kelly_fraction(0.6, 2.0, fraction=0.25) == pytest.approx(0.05)
    assert kelly_fraction(0.4, 2.0) == 0.0

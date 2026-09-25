import numpy as np
import pandas as pd

from edgeline import db
from edgeline.alerts import format_message, pending_alerts, send
from edgeline.models.banlist import evaluate
from edgeline.models.copula import Component, build_corr, joint_hit_probability, simulate, sum_over_under_push


def _comps():
    return [
        Component(5.0, "A", "T1", 1), Component(5.0, "A", "T1", 2),  # same player, two maps
        Component(4.0, "B", "T1", 1),  # teammate of A on map 1
        Component(6.0, "C", "T2", 1),  # opponent on map 1
    ]


def test_build_corr_structure():
    C = build_corr(_comps(), rho_self=0.2, rho_team=0.3, rho_opp=-0.25)
    assert np.allclose([C[0, 1], C[0, 2], C[0, 3]], [0.2, 0.3, -0.25], atol=1e-6)
    assert np.isclose(C[1, 2], 0.3 * 0.2)  # teammate, different map: attenuated
    assert np.allclose(np.diag(C), 1.0)
    assert np.all(np.linalg.eigvalsh(C) > 0)


def test_simulate_preserves_marginals_and_induces_correlation():
    comps = _comps()
    C = build_corr(comps, 0.3, 0.3, -0.3)
    X = simulate(comps, r=8.0, corr=C, n=60000, rng=np.random.default_rng(1))
    assert np.allclose(X.mean(axis=0), [5, 5, 4, 6], rtol=0.03)
    c = np.corrcoef(X.T)
    assert 0.15 < c[0, 1] < 0.35
    assert 0.15 < c[0, 2] < 0.35
    assert -0.35 < c[0, 3] < -0.15


def test_sum_over_under_push_sums_to_one():
    comps = _comps()[:2]
    o, u, p = sum_over_under_push(9.5, comps, 8.0, 0.2, 0.0, 0.0)
    assert abs(o + u + p - 1) < 1e-9 and p == 0
    assert 0.4 < o < 0.7


def test_joint_probability_reflects_dependence():
    comps = _comps()
    legs = [([comps[0]], 4.5, "OVER"), ([comps[3]], 5.5, "OVER")]  # opponents, both over
    j_neg, (pa, pb) = joint_hit_probability(legs, 8.0, 0.0, 0.0, -0.4)
    j_pos, _ = joint_hit_probability(legs, 8.0, 0.0, 0.0, 0.4)
    assert j_neg < pa * pb < j_pos


def test_banlist_flags_significant_underperformers():
    rows = []
    rows += [{"sport": "dota", "player_name": "bad", "prob": 0.6, "hit": 1 if i < 8 else 0} for i in range(30)]   # 8/30 vs 60%
    rows += [{"sport": "dota", "player_name": "good", "prob": 0.6, "hit": 1 if i < 19 else 0} for i in range(30)]  # 19/30
    rows += [{"sport": "dota", "player_name": "few", "prob": 0.6, "hit": 0} for _ in range(5)]                   # too few
    t = evaluate(pd.DataFrame(rows), min_n=15, alpha=0.05).set_index("player_name")
    assert t.loc["bad", "banned"] == 1
    assert t.loc["good", "banned"] == 0
    assert t.loc["few", "banned"] == 0


def test_alerts_pending_and_format(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    conn.execute("INSERT INTO lines(book, projection_id, sport, player_name, team, opponent, stat_type, board_time, start_time, open_line, open_seen_at, current_line, last_seen_at) VALUES ('prizepicks','1','dota','P','A','B','MAP 1 Kills','t','t',5.5,'t',5.5,'t')")
    conn.execute("INSERT INTO predictions(book, projection_id, model_version, computed_at, line, projection, p_over, p_under, ev_over, ev_under, lean, prob, ev, bettable, notes) VALUES ('prizepicks','1','v','t',5.5,7.0,0.7,0.3,0.24,-0.5,'OVER',0.7,0.24,1,'')")
    conn.commit()
    df = pending_alerts(conn)
    assert len(df) == 1
    msg = format_message(df)
    assert "P (A vs B) MAP 1 Kills **OVER 5.5**" in msg and "projections=1-o-5.5" in msg
    assert send(conn, dry_run=True) == 1
    assert len(pending_alerts(conn)) == 1  # dry run does not mark as sent
    assert send(conn, webhook=None) == 0  # no webhook configured: nothing sent, nothing marked
    assert len(pending_alerts(conn)) == 1


def test_logit_calibrator_is_smooth_and_monotone():
    from edgeline.models.props import LogitCalibrator

    rng = np.random.default_rng(0)
    raw = rng.uniform(0.05, 0.95, 20000)
    obs = (rng.uniform(size=raw.size) < np.clip(0.5 + 1.3 * (raw - 0.5), 0, 1)).astype(float)  # over-confident-ish truth
    c = LogitCalibrator.fit(raw, obs)
    p = c.predict(np.array([0.05, 0.3, 0.5, 0.7, 0.95]))
    assert np.all(np.diff(p) > 0) and 0.0 < p[0] < 0.2 and 0.8 < p[-1] < 1.0


def test_name_resolver_strips_tags():
    from edgeline.models.predict import NameResolver

    r = NameResolver(["NAF", "s1mple", "Staehr"], ["Team Liquid"])
    assert r.player("NAF-FLY") == "NAF"
    assert r.player("S1MPLE") == "s1mple"
    assert r.player("unknown-guy") is None


def test_name_resolver_sponsor_tags_and_rosters():
    from edgeline.models.predict import NameResolver

    hist = ["Eros//Cryptic", "R5 ||SLIGHT", "TP |ogwizard", "WUMBO1", "1corim", "mesaminicx", "grizzly", "nicxo", "b1t"]
    rosters = {"Marsborne": ["TP |ogwizard", "WUMBO1", "mesaminicx", "grizzly"], "Other": ["nicxo"], "Overtake Sector": ["1corim"]}
    r = NameResolver(hist, ["Marsborne", "Other"], rosters)
    # sponsor tags and edge digits are stripped on the history side
    assert r.player("Cryptic") == "Eros//Cryptic"
    assert r.player("SLIGHT") == "R5 ||SLIGHT"
    assert r.player("ogwizard") == "TP |ogwizard"
    assert r.player("WUMBO") == "WUMBO1"
    assert r.player("corim") == "1corim"
    assert r.player("b1t") == "b1t"
    # containment works only inside the given roster and only when unique
    assert r.player("nicx", "Marsborne") == "mesaminicx"
    assert r.player("Grizz", "Marsborne") == "grizzly"
    assert r.player("nicx") is None
    assert r.player("intra", "Marsborne") is None
    assert r.player("nic", "Marsborne") is None
    # a roster match beats a same-named player on another team
    r2 = NameResolver(["Jason", "Eros//jason"], ["A", "B"], {"A": ["Jason"], "B": ["Eros//jason"]})
    assert r2.player("Jason", "B") == "Eros//jason"
    assert r2.player("Jason", "A") == "Jason"
    assert r2.player("Jason") == "Jason"


def test_stacks_skip_pairs_sharing_a_player():
    from itertools import combinations

    from edgeline.models.predict import PricedLine

    def leg(pid, players):
        comps = [Component(10.0, p, "T", 1) for p in players]
        return PricedLine(pid, "g", "kills", 20.5, comps, 20.0, 0.6, 0.4, 0.0, 0.6, "OVER", 0.6, 0.1, True)

    ps = [leg("combo", ["A", "B"]), leg("a", ["A"]), leg("c", ["C"])]
    pairs = [
        (a, b) for a, b in combinations(ps, 2)
        if a.stat == b.stat and not ({c.player for c in a.components} & {c.player for c in b.components})
    ]
    assert {(a.projection_id, b.projection_id) for a, b in pairs} == {("combo", "c"), ("a", "c")}


def test_lolesports_helpers():
    import datetime as dt

    from edgeline.data.lolesports import _round10

    assert _round10(dt.datetime(2026, 9, 13, 5, 0, 17)) == "2026-09-13T05:00:10Z"


def test_cod_map_mode_override_reaches_features():
    from edgeline.features.build import assemble_prediction_row
    from edgeline.models.predict import COD_MODE_BY_MAP

    prow = pd.Series({"role": "HP", "league": "x", "tier": "unknown", "role_state": {"HP": {"pr_kills_mean10": 25.0, "pr_games": 9}, "S&D": {"pr_kills_mean10": 6.0, "pr_games": 9}}})
    hp = assemble_prediction_row(prow, None, None, 1, 0, None, COD_MODE_BY_MAP[1])
    snd = assemble_prediction_row(prow, None, None, 2, 0, None, COD_MODE_BY_MAP[2])
    assert hp["role"] == "HP" and hp["pr_kills_mean10"] == 25.0
    assert snd["role"] == "S&D" and snd["pr_kills_mean10"] == 6.0


def test_unproven_markets_are_gated():
    from edgeline.models.predict import market_is_unproven

    assert market_is_unproven("cs2", "kills") and market_is_unproven("cod", "kills")
    assert not market_is_unproven("cs2", "headshots") and not market_is_unproven("dota", "kills") and not market_is_unproven("lol", "kills")

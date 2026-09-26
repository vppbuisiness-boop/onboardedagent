import numpy as np
import pandas as pd

from edgeline.features.build import build_training_frame, current_state


def _history() -> pd.DataFrame:
    rows = []
    kills = {"A": [3, 5, 7, 9], "B": [10, 8, 6, 4]}
    for i in range(4):
        for team, opp, players in (("T1", "T2", ["A"]), ("T2", "T1", ["B"])):
            for p in players:
                rows.append(
                    dict(sport="x", source="t", game_id=f"g{i}", series_id=f"s{i//2}", game_number=i % 2 + 1,
                         date=f"2026-01-0{i+1}T00:00:00Z", league="L", tier=None, patch=None, player_name=p, player_id=p,
                         team=team, opponent=opp, role="mid", side="blue", champion=None, kills=kills[p][i], deaths=2, assists=4,
                         headshots=None, team_kills=kills[p][i] + 5, opp_kills=kills["B" if p == "A" else "A"][i] + 5,
                         game_length=30.0, rounds=[20, 24, 16, 30][i], adr=[60.0, 80.0, 70.0, 90.0][i], win=int(p == "A"), playoffs=0)
                )
    return pd.DataFrame(rows)


def test_features_use_only_prior_games():
    frame = build_training_frame(_history())
    a = frame[frame.player_name == "A"].sort_values("date")
    # first game has no prior info
    assert np.isnan(a.iloc[0]["p_kills_mean10"])
    # second game sees only game 1 (3 kills), third sees mean(3,5)=4
    assert a.iloc[1]["p_kills_mean10"] == 3
    assert a.iloc[2]["p_kills_mean10"] == 4
    assert a.iloc[3]["p_kills_mean5"] == 5
    assert a.iloc[1]["p_games"] == 1
    # opponent feature at game 3 for A = B's team conceded kills before game 3: A's team_kills in g0,g1 = 8, 10 -> 9
    assert a.iloc[2]["o_conceded_mean10"] == 9


def test_current_state_includes_latest_game():
    player_state, team_state = current_state(_history())
    assert player_state.loc["A", "p_kills_mean10"] == 6  # mean(3,5,7,9)
    assert player_state.loc["A", "p_games"] == 4
    assert team_state.loc["T1", "t_win10"] == 1.0
    assert team_state.loc["T2", "t_kills_mean10"] == np.mean([15, 13, 11, 9])
    # evidence depth: games with the latest team inside the last 60 days as of the given time
    fresh, _ = current_state(_history(), asof=pd.Timestamp("2026-01-10T00:00:00Z"))
    assert fresh.loc["A", "p_team_games60"] == 4 and abs(fresh.loc["A", "p_days_since"] - 6.0) < 1e-9
    stale, _ = current_state(_history(), asof=pd.Timestamp("2026-06-01T00:00:00Z"))
    assert stale.loc["A", "p_team_games60"] == 0


def test_elo_and_role_matchup_features_are_as_of():
    from edgeline.features.build import ELO_START, build_training_frame

    frame = build_training_frame(_history())
    a = frame[frame.player_name == "A"].sort_values("date")
    # first game: both teams at the start rating; A's team wins every game so its Elo rises afterwards
    assert a.iloc[0]["t_elo"] == ELO_START and a.iloc[0]["o_elo"] == ELO_START
    assert a.iloc[1]["t_elo"] > ELO_START > a.iloc[1]["o_elo"]
    assert 0.5 < a.iloc[1]["p_win_elo"] < 1.0
    # opponent role matchup for A at game 3: B (mid, team T2) scored 10, 8 before -> 9; T2 conceded to mid 3, 5 -> 4
    assert a.iloc[2]["o_role_kills10"] == 9
    assert a.iloc[2]["o_role_conceded10"] == 4
    assert np.isnan(a.iloc[0]["o_role_kills10"])


def test_roi_gauge_math():
    from edgeline.grading.roi import n_needed, parlay_roi, wilson

    assert abs(parlay_roi(0.6) - 0.296) < 1e-6
    p, lo, hi = wilson(60, 100)
    assert abs(p - 0.6) < 1e-9 and lo < 0.6 < hi and 0.49 < lo < 0.51
    assert n_needed(0.55, 0.5623) is None
    assert n_needed(0.65, 0.5623) < n_needed(0.60, 0.5623)


def test_round_rate_features_are_as_of_and_prediction_row_matches():
    from edgeline.features.build import assemble_prediction_row, build_training_frame, current_state

    frame = build_training_frame(_history())
    a = frame[frame.player_name == "A"].sort_values("date")
    # per-round kill rate before game 3 = mean(3/20, 5/24); rounds seen so far = mean(20, 24)
    assert abs(a.iloc[2]["p_kills_pr10"] - np.mean([3 / 20, 5 / 24])) < 1e-9
    assert a.iloc[2]["p_rounds_mean10"] == 22
    assert np.isnan(a.iloc[0]["p_kills_pr10"])
    # both teams played the same maps, so expected rounds = mean of the two rolling means = 22
    assert a.iloc[2]["t_rounds_mean10"] == 22 and a.iloc[2]["o_rounds_mean10"] == 22 and a.iloc[2]["exp_rounds"] == 22
    assert abs(a.iloc[2]["kills_pr_x_rounds"] - a.iloc[2]["p_kills_pr10"] * 22) < 1e-9
    player_state, team_state = current_state(_history())
    row = assemble_prediction_row(player_state.loc["A"], team_state.loc["T1"], team_state.loc["T2"], 1, 0, None, None)
    assert row["exp_rounds"] == np.mean([20, 24, 16, 30])
    assert abs(row["kills_pr_x_rounds"] - player_state.loc["A", "p_kills_pr10"] * row["exp_rounds"]) < 1e-9
    # off by default: the CS2 backtest showed no gain
    from edgeline.features.build import FEATURE_COLUMNS, USE_ROUND_FEATURES

    assert USE_ROUND_FEATURES or ("exp_rounds" not in FEATURE_COLUMNS and "t_rounds_mean10" not in FEATURE_COLUMNS)


def test_canonical_teams_merges_case_variants():
    from edgeline.features.build import canonical_teams

    df = pd.DataFrame({"team": ["fnatic", "FNATIC", "fnatic", "Fnatic", "ENCE"], "opponent": ["ENCE", "ENCE", "Fnatic", "fnatic", "FNATIC"]})
    out = canonical_teams(df)
    assert out["team"].tolist() == ["fnatic"] * 4 + ["ENCE"]
    assert out["opponent"].tolist() == ["ENCE", "ENCE", "fnatic", "fnatic", "fnatic"]


def test_extra_stat_features_are_as_of_and_gated():
    from edgeline.features.build import EXTRA_STAT_FEATURES, FEATURE_COLUMNS, USE_EXTRA_STATS, assemble_prediction_row, build_training_frame, current_state

    frame = build_training_frame(_history())
    a = frame[frame.player_name == "A"].sort_values("date")
    assert np.isnan(a.iloc[0]["p_adr_mean10"]) and a.iloc[2]["p_adr_mean10"] == 70.0  # mean(60, 80)
    assert np.isnan(a.iloc[1]["p_kast_mean10"])  # not in the fixture -> NaN, never an error
    player_state, team_state = current_state(_history())
    row = assemble_prediction_row(player_state.loc["A"], team_state.loc["T1"], team_state.loc["T2"], 1, 0, None, None)
    assert row["p_adr_mean10"] == 75.0
    assert USE_EXTRA_STATS or all(c not in FEATURE_COLUMNS for c in EXTRA_STAT_FEATURES)


def test_fair_line_sits_at_the_median_not_the_mean():
    from edgeline.models.distributions import fair_line, over_under_push

    r = 50.0  # near-Poisson
    lines = fair_line(np.array([4.0, 4.3, 4.8, 0.2]), r)
    assert list(lines) == [3.5, 4.5, 4.5, 0.5]
    # the chosen line has the over probability closest to 50% of the two candidates
    for m, l in zip([4.0, 4.3, 4.8], lines[:3]):
        p_l = over_under_push(l, m, r)[0]
        other = l + 1 if l < np.floor(m) else l - 1
        assert abs(p_l - 0.5) <= abs(over_under_push(other, m, r)[0] - 0.5)


def test_mean_bias_uses_the_recent_window():
    from edgeline.models.props import _mean_bias

    dates = pd.date_range("2026-01-01", periods=400, freq="12h", tz="UTC").to_numpy()
    mu = np.full(400, 10.0)
    y = np.where(np.arange(400) < 219, 9.5, 10.5)  # old rows run low, the last 90 days (181 half-day rows) run high
    assert abs(_mean_bias(dates, y, mu, window_days=90, min_rows=50) - 1.05) < 1e-9
    assert abs(_mean_bias(dates, y, mu, window_days=90, min_rows=10_000) - 1.0) < 0.01  # thin window -> whole split
    assert _mean_bias(dates, y * 5, mu) == 1.1  # clipped


def test_two_map_pairs_use_map1_features_and_sum_both_maps():
    from edgeline.models.backtest import two_map_pairs

    df = pd.DataFrame({"series_id": ["s1", "s1", "s1", "s2"], "player_name": ["A", "A", "B", "A"], "game_number": [1, 2, 1, 1],
                       "kills": [3, 5, 7, 9], "p_kills_mean10": [2.0, 2.5, 6.0, 4.0], "date": pd.to_datetime(["2026-01-01"] * 4, utc=True)})
    out = two_map_pairs(df, "kills")
    assert len(out) == 1 and out.iloc[0]["player_name"] == "A"
    assert out.iloc[0]["actual_sum"] == 8 and out.iloc[0]["p_kills_mean10"] == 2.0  # map-1 features, both maps summed

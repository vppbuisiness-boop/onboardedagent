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
                         game_length=30.0, win=int(p == "A"), playoffs=0)
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

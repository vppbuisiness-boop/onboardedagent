import json

import pandas as pd

from edgeline import db
from edgeline.books.prizepicks import LineRecord, upsert_lines
from edgeline.grading.grade import grade_lines
from edgeline.slips.builder import build, mark_used



PG_COLS = ["sport", "source", "game_id", "series_id", "game_number", "date", "league", "tier", "patch", "player_name", "player_id", "team",
           "opponent", "role", "side", "champion", "kills", "deaths", "assists", "headshots", "team_kills", "opp_kills", "game_length", "rounds",
           "win", "playoffs"]


def _insert_games(conn, rows):
    conn.executemany(f"INSERT INTO player_games({','.join(PG_COLS)}) VALUES ({','.join('?' * len(PG_COLS))})", rows)

def _line(pid: str, name: str, team: str, opp: str, game: str, stat_type: str, line: float, start: str, map_to: int = 1) -> LineRecord:
    return LineRecord("prizepicks", pid, "dota", "Dota2", pid, name, team, opp, None, game, stat_type, "kills", 1, map_to, 0, None,
                      int(map_to == 3), start, start, line, "standard", "pre_game", start)


def _seed(conn):
    recs = [
        _line("1", "P1", "TA", "TB", "gA", "MAP 1 Kills", 5.5, "2099-01-01T10:00:00Z"),
        _line("2", "P2", "TA", "TB", "gA", "MAP 1 Kills", 6.5, "2099-01-01T10:00:00Z"),
        _line("3", "P3", "TC", "TD", "gB", "MAP 1 Kills", 7.5, "2099-01-01T12:00:00Z"),
        _line("4", "P4", "TE", "TF", "gC", "MAPS 1-3 Kills", 20.5, "2099-01-01T14:00:00Z", map_to=3),
    ]
    upsert_lines(conn, recs, "2026-01-01T00:00:00Z")
    preds = [("1", 0.70, 0.245), ("2", 0.65, 0.156), ("3", 0.62, 0.10), ("4", 0.61, 0.08)]
    for pid, p, ev in preds:
        conn.execute(
            "INSERT INTO predictions(book, projection_id, model_version, computed_at, line, projection, p_over, p_under, ev_over, ev_under, lean, prob, ev, bettable, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("prizepicks", pid, "v1", "2026-01-01T01:00:00Z", 5.5, 6.0, p, 1 - p, ev, -0.5, "OVER", p, ev, 1, ""),
        )
    conn.commit()


def test_open_line_tracking(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    upsert_lines(conn, [_line("9", "X", "A", "B", "g", "MAP 1 Kills", 4.5, "2099-01-01T10:00:00Z")], "2026-01-01T00:00:00Z")
    upsert_lines(conn, [_line("9", "X", "A", "B", "g", "MAP 1 Kills", 5.5, "2099-01-01T10:00:00Z")], "2026-01-01T00:05:00Z")
    row = conn.execute("SELECT open_line, current_line FROM lines WHERE projection_id='9'").fetchone()
    assert (row["open_line"], row["current_line"]) == (4.5, 5.5)
    assert conn.execute("SELECT COUNT(*) FROM line_snapshots").fetchone()[0] == 2


def test_builder_respects_one_leg_per_game_and_used_marks(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    _seed(conn)
    slips = build(conn, "prizepicks", "POWER", 3, max_slips=5)
    assert len(slips) == 1
    ids = {l["projection_id"] for l in slips[0].legs}
    assert ids == {"1", "3", "4"}  # 2 shares game gA with 1
    assert slips[0].ev > 0 and 0 < slips[0].hit_prob < 1
    assert slips[0].link.startswith("https://app.prizepicks.com/?projections=1-o-5.5")
    mark_used(conn, "prizepicks", ["1"])
    slips2 = build(conn, "prizepicks", "POWER", 3, max_slips=5)
    assert {l["projection_id"] for l in slips2[0].legs} == {"2", "3", "4"}


def test_grading_sums_maps_and_voids_short_series(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    _seed(conn)
    rows = []
    # P1 plays one map: 7 kills -> over 5.5. P4's series ends 2-0 -> MAPS 1-3 void.
    rows.append(("dota", "t", "m1", "s1", 1, "2099-01-01T10:30:00Z", "L", None, None, "P1", "P1", "TA", "TB", "mid", "r", None, 7, 1, 1, None, 20, 10, 30.0, None, 1, 0))
    rows.append(("dota", "t", "m2", "s2", 1, "2099-01-01T14:30:00Z", "L", None, None, "P4", "P4", "TE", "TF", "mid", "r", None, 9, 1, 1, None, 20, 10, 30.0, None, 1, 0))
    rows.append(("dota", "t", "m3", "s2", 2, "2099-01-01T15:30:00Z", "L", None, None, "P4", "P4", "TE", "TF", "mid", "r", None, 8, 1, 1, None, 20, 10, 30.0, None, 1, 0))
    _insert_games(conn, rows)
    conn.commit()
    out = grade_lines(conn, "dota", "prizepicks", min_age_hours=-10**6, settle_hours=-10**6)  # lines are dated in 2099; force them due and settled
    assert out["graded"] == 2 and out["void"] == 1
    g = {r["projection_id"]: r for r in conn.execute("SELECT * FROM grades").fetchall()}
    assert g["1"]["result_open"] == "over" and g["1"]["actual"] == 7
    assert g["4"]["result_open"] == "void" and g["4"]["maps_played"] == 2


def test_builder_skips_started_games(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    _seed(conn)
    conn.execute("UPDATE lines SET start_time='2000-01-01T00:00:00Z' WHERE projection_id='1'")
    conn.commit()
    slips = build(conn, "prizepicks", "POWER", 3, max_slips=5)
    assert "1" not in {l["projection_id"] for l in slips[0].legs}


def test_grading_leaves_partial_series_pending():
    from edgeline.grading.grade import grade_lines

    conn = db.connect(":memory:")
    _seed(conn)
    # P4's series has only map 1 loaded (1-0): a MAPS 1-3 line must stay pending, not void
    rows = [("dota", "t", "m2", "s2", 1, "2099-01-01T14:30:00Z", "L", None, None, "P4", "P4", "TE", "TF", "mid", "r", None, 9, 1, 1, None, 20, 10, 30.0, None, 1, 0)]
    _insert_games(conn, rows)
    conn.commit()
    out = grade_lines(conn, "dota", "prizepicks", min_age_hours=-10**6, settle_hours=-10**6)
    graded = {r[0]: r[1] for r in conn.execute("SELECT projection_id, result_open FROM grades")}
    assert "4" not in graded and out["void"] == 0


def test_clv_grades_open_to_close_against_projection_side(tmp_path):
    from edgeline.grading.clv import clv_frame, summarize

    conn = db.connect(tmp_path / "t.db")
    start = "2026-01-02T10:00:00Z"
    # line A opens 5.5, closes 4.5 (moved down); line B opens 5.5, closes 6.5 (moved up); line C never moves
    upsert_lines(conn, [_line("A", "P1", "TA", "TB", "gA", "MAP 1 Kills", 5.5, start),
                        _line("B", "P2", "TA", "TB", "gA", "MAP 1 Kills", 5.5, start),
                        _line("C", "P3", "TC", "TD", "gB", "MAP 1 Kills", 7.5, start)], "2026-01-01T00:00:00Z")
    upsert_lines(conn, [_line("A", "P1", "TA", "TB", "gA", "MAP 1 Kills", 4.5, start),
                        _line("B", "P2", "TA", "TB", "gA", "MAP 1 Kills", 6.5, start),
                        _line("C", "P3", "TC", "TD", "gB", "MAP 1 Kills", 7.5, start)], "2026-01-01T06:00:00Z")
    # a snapshot after the start must not count as the close
    upsert_lines(conn, [_line("A", "P1", "TA", "TB", "gA", "MAP 1 Kills", 9.5, start)], "2026-01-02T11:00:00Z")
    for pid, proj in (("A", 4.0), ("B", 4.0), ("C", 9.0)):  # A: UNDER at the open (for), B: UNDER (against), C: OVER, unchanged
        conn.execute(
            "INSERT INTO predictions(book, projection_id, model_version, computed_at, line, projection, p_over, p_under, ev_over, ev_under, lean, prob, ev, bettable, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("prizepicks", pid, "v1", "2026-01-01T07:00:00Z", 5.5, proj, 0.4, 0.6, -0.1, 0.05, "UNDER", 0.6, 0.05, 1, ""),
        )
    conn.commit()
    df = clv_frame(conn, "prizepicks", now=pd.Timestamp("2026-01-03", tz="UTC")).set_index("projection_id")
    assert df.loc["A", "close_line"] == 4.5 and df.loc["A", "clv"] == 1.0
    assert df.loc["B", "close_line"] == 6.5 and df.loc["B", "clv"] == -1.0
    assert df.loc["C", "lean_open"] == "OVER" and df.loc["C", "clv"] == 0.0
    s = summarize(df.reset_index()).iloc[0]
    assert (s["n"], s["unchanged"], s["for"], s["against"]) == (3, 1, 1, 1)
    # a game that has not started yet is excluded
    assert clv_frame(conn, "prizepicks", now=pd.Timestamp("2026-01-01T12:00:00Z")).empty


def test_kalshi_grade_settles_map_markets_from_history(tmp_path):
    from edgeline.books.kalshi import SCHEMA, event_time, grade, summarize_grades

    assert event_time("KXCS2MAP-26SEP261600ABCD-2-AB") == pd.Timestamp("2026-09-26T16:00:00Z")
    assert event_time("nonsense") is None
    conn = db.connect(tmp_path / "t.db")
    conn.executescript(SCHEMA)
    base = ["cs2", "test", None, "s1", None, None, "L", 1, None, None, None, None, None, None, None, None, 10, 5, 1, 3, 40, 30, 1800, 24, None, 0]
    rows = []
    for game, gn, team, opp, win in (("g1", 1, "TA", "TB", 1), ("g1", 1, "TB", "TA", 0), ("g2", 2, "TA", "TB", 0), ("g2", 2, "TB", "TA", 1)):
        r = dict(zip(PG_COLS, base)); r.update(game_id=game, game_number=gn, date="2026-09-26T16:40:00Z", player_name=f"p_{team}", player_id=f"p_{team}", team=team, opponent=opp, win=win)
        rows.append([r[c] for c in PG_COLS])
    _insert_games(conn, rows)
    quotes = [("KXCS2MAP-26SEP261600TATB-1-TA", "2026-09-26T15:00:00+00:00", "e1", "cs2", "TA", "TB", 1, "map", None, 0.50, 0.56, None, 0.70),
              ("KXCS2MAP-26SEP261600TATB-2-TA", "2026-09-26T15:00:00+00:00", "e2", "cs2", "TA", "TB", 2, "map", None, 0.60, 0.66, None, 0.70),
              ("KXCS2MAP-26SEP261600TATB-3-TA", "2026-09-26T15:00:00+00:00", "e3", "cs2", "TA", "TB", 3, "map", None, 0.50, 0.56, None, 0.70)]
    conn.executemany("INSERT INTO kalshi_quotes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", quotes)
    conn.commit()
    g = grade(conn, min_age_hours=3.0, now=pd.Timestamp("2026-09-27T00:00:00Z"))
    assert g.attrs["new"] == 2 and set(g["ticker"]) == {"KXCS2MAP-26SEP261600TATB-1-TA", "KXCS2MAP-26SEP261600TATB-2-TA"}  # map 3 never played
    assert g.set_index("map_index")["actual"].to_dict() == {1: 1, 2: 0}
    s = summarize_grades(g).set_index("sport").loc["cs2"]
    assert s["n"] == 2 and s["trades@0.10"] == 1  # only map 1 clears a 10c edge at the ask (0.70 - 0.56)
    assert abs(s["roi@0.10"] - (1 - 0.56) / 0.56) < 1e-9
    assert grade(conn, now=pd.Timestamp("2026-09-27T00:00:00Z")).attrs["new"] == 0  # idempotent

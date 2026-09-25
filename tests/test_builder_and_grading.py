import json

import pandas as pd

from edgeline import db
from edgeline.books.prizepicks import LineRecord, upsert_lines
from edgeline.grading.grade import grade_lines
from edgeline.slips.builder import build, mark_used


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
    rows.append(("dota", "t", "m1", "s1", 1, "2099-01-01T10:30:00Z", "L", None, None, "P1", "P1", "TA", "TB", "mid", "r", None, 7, 1, 1, None, 20, 10, 30.0, 1, 0))
    rows.append(("dota", "t", "m2", "s2", 1, "2099-01-01T14:30:00Z", "L", None, None, "P4", "P4", "TE", "TF", "mid", "r", None, 9, 1, 1, None, 20, 10, 30.0, 1, 0))
    rows.append(("dota", "t", "m3", "s2", 2, "2099-01-01T15:30:00Z", "L", None, None, "P4", "P4", "TE", "TF", "mid", "r", None, 8, 1, 1, None, 20, 10, 30.0, 1, 0))
    conn.executemany("INSERT INTO player_games VALUES (" + ",".join("?" * 25) + ")", rows)
    conn.commit()
    out = grade_lines(conn, "dota", "prizepicks", min_age_hours=-10**6)  # lines are dated in 2099; force them due
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

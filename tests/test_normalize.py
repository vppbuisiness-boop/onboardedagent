from edgeline.books.normalize import PropScope, opponent_from_game_id, parse_stat_type, voidable


def test_parse_single_map():
    s = parse_stat_type("MAP 1 Kills")
    assert s == PropScope("kills", 1, 1, False)
    assert s.n_maps == 1


def test_parse_map_range_and_combo():
    s = parse_stat_type("MAPS 1-3 Kills (Combo)")
    assert s.stat == "kills" and s.map_from == 1 and s.map_to == 3 and s.combo


def test_parse_headshots_and_unknown():
    assert parse_stat_type("MAPS 1-2 Headshots").stat == "headshots"
    assert parse_stat_type("Fantasy Score") is None


def test_opponent_from_game_id():
    assert opponent_from_game_id("fnaticHEROIC46290.1666666667", "fnatic") == "HEROIC"
    assert opponent_from_game_id("fnaticHEROIC46290.1666666667", "HEROIC") == "fnatic"
    assert opponent_from_game_id("ex-RUBYSINNERS46290.4583333333", "SINNERS") == "RUBY"
    assert opponent_from_game_id("Cupid EsportsDPKC46290.0833333333", "Cupid Esports") == "DPKC"
    assert opponent_from_game_id("XYZABC1.0", "QQQ") is None


def test_voidable_rules():
    assert not voidable("lol", parse_stat_type("MAP 1 Kills"))
    assert not voidable("lol", parse_stat_type("MAPS 1-2 Kills"))
    assert voidable("lol", parse_stat_type("MAPS 1-3 Kills"))  # BO3 assumed
    assert not voidable("lol", parse_stat_type("MAPS 1-3 Kills"), series_format=5)

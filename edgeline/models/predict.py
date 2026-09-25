"""Price current book lines with trained models: projection, P(over/under), EV, bettable flag."""
from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3

import numpy as np
import pandas as pd

from ..config import DEFAULT_MAX_LINE_MOVE, DEFAULT_MIN_EV, DEFAULT_MIN_PROB
from ..ev.payouts import leg_decimal_odds
from ..features.build import assemble_prediction_row, current_state
from .distributions import over_under_push, sum_over_under_push
from .props import PropModel

SUPPORTED_STATS = ("kills", "deaths", "assists")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


class NameResolver:
    """Map book player/team strings onto history names (exact, normalized, initials)."""

    def __init__(self, players: list[str], teams: list[str]):
        self.players = {p: p for p in players}
        self.players_norm = {}
        for p in players:
            self.players_norm.setdefault(_norm(p), p)
        self.teams = {t: t for t in teams}
        self.teams_norm = {_norm(t): t for t in teams}
        self.team_initials = {}
        for t in teams:
            words = re.findall(r"[A-Za-z0-9]+", t)
            if len(words) >= 2:
                self.team_initials.setdefault("".join(w[0] for w in words).upper(), t)

    def player(self, name: str) -> str | None:
        if name in self.players:
            return name
        return self.players_norm.get(_norm(name))

    def team(self, code: str | None) -> str | None:
        if not code:
            return None
        if code in self.teams:
            return code
        n = _norm(code)
        if n in self.teams_norm:
            return self.teams_norm[n]
        if code.upper() in self.team_initials:
            return self.team_initials[code.upper()]
        # prefix match on a single candidate
        cands = [t for k, t in self.teams_norm.items() if k.startswith(n)] if len(n) >= 3 else []
        return cands[0] if len(cands) == 1 else None


def resolve_board_teams(lines: pd.DataFrame, player_state: pd.DataFrame, resolver: "NameResolver") -> dict[str, str]:
    """Map each book team code to a history team name by majority vote of the code's players' latest teams.

    Falls back to string matching only for codes with no resolvable players.
    """
    votes: dict[str, dict[str, int]] = {}
    for code, name in zip(lines["team"], lines["player_name"]):
        if not code or not isinstance(name, str) or "+" in name:
            continue
        p = resolver.player(name)
        if p is None or p not in player_state.index:
            continue
        team = player_state.loc[p].get("team")
        if isinstance(team, str) and team:
            votes.setdefault(code, {})
            votes[code][team] = votes[code].get(team, 0) + 1
    out = {code: max(v.items(), key=lambda kv: kv[1])[0] for code, v in votes.items()}
    codes = set(lines["team"].dropna()) | set(lines["opponent"].dropna())
    for code in codes:
        if code not in out:
            t = resolver.team(code)
            if t:
                out[code] = t
    return out


def load_models(sport: str) -> dict[str, PropModel]:
    out = {}
    for stat in SUPPORTED_STATS:
        try:
            out[stat] = PropModel.load(sport, stat)
        except FileNotFoundError:
            continue
    return out


def price_line(model: PropModel, feats_by_map: list[dict], line: float, groups: list[int] | None = None) -> tuple[float, float, float, float]:
    """Return (projection, p_over, p_under, p_push) for a line over the given map feature rows."""
    X = pd.DataFrame(feats_by_map)
    mus = model.predict_mu(X)
    proj = float(mus.sum())
    if len(mus) == 1:
        over, under, push = over_under_push(line, float(mus[0]), model.r)
    else:
        over, under, push = sum_over_under_push(line, mus.tolist(), model.r, model.phi, groups=groups)
    over_c = model.calibrate(over)
    # keep push mass; rescale under so the three sum to 1
    under_c = max(0.0, 1.0 - over_c - push)
    return proj, float(over_c), float(under_c), float(push)


def price_board(conn: sqlite3.Connection, sport: str, book: str = "prizepicks", min_prob: float = DEFAULT_MIN_PROB,
                min_ev: float = DEFAULT_MIN_EV, max_move: float = DEFAULT_MAX_LINE_MOVE, skip_voidable: bool = True,
                only_upcoming: bool = True) -> pd.DataFrame:
    models = load_models(sport)
    if not models:
        raise FileNotFoundError(f"no trained models for sport={sport}; run `edgeline model train --sport {sport}`")
    pg = pd.read_sql_query("SELECT * FROM player_games WHERE sport=?", conn, params=(sport,))
    if pg.empty:
        raise RuntimeError(f"no history rows for sport={sport}")
    player_state, team_state = current_state(pg)
    resolver = NameResolver(list(player_state.index), list(team_state.index))
    odds = leg_decimal_odds(book)

    q = "SELECT * FROM lines WHERE book=? AND sport=?"
    lines = pd.read_sql_query(q, conn, params=(book, sport))
    if only_upcoming:
        now = pd.Timestamp.now(tz="UTC")
        st = pd.to_datetime(lines["start_time"], utc=True, errors="coerce")
        lines = lines[(st.isna()) | (st > now - pd.Timedelta(hours=1))]
    team_map = resolve_board_teams(lines, player_state, resolver)
    computed_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rows = []
    for ln in lines.itertuples(index=False):
        note = []
        stat = ln.stat
        if stat not in models:
            continue
        model = models[stat]
        names = json.loads(ln.combo_players) if ln.combo and ln.combo_players else [ln.player_name]
        resolved = [resolver.player(n) for n in names]
        if any(r is None for r in resolved):
            note.append("player_unmapped:" + ",".join(n for n, r in zip(names, resolved) if r is None))
            rows.append(_row(ln, model.version, computed_at, None, None, None, None, None, None, None, 0, note))
            continue
        team_name = team_map.get(ln.team) or player_state.loc[resolved[0]].get("team")
        opp_name = team_map.get(ln.opponent)
        if opp_name is None:
            note.append("opponent_unmapped")
        team_row = team_state.loc[team_name] if team_name in team_state.index else None
        opp_row = team_state.loc[opp_name] if opp_name in team_state.index else None
        maps = list(range(int(ln.map_from), int(ln.map_to) + 1))
        feats, groups = [], []
        for pi, pname in enumerate(resolved):
            prow = player_state.loc[pname]
            for m in maps:
                feats.append(assemble_prediction_row(prow, team_row, opp_row, m, 0, None, None))
                groups.append(0)  # same game: shared frailty
        proj, p_over, p_under, p_push = price_line(model, feats, float(ln.current_line), groups)
        # Pushes refund the stake: EV = P(win) * (odds - 1) - P(lose), with P(lose) = 1 - P(win) - P(push).
        ev_over = p_over * (odds - 1) - (1 - p_over - p_push)
        ev_under = p_under * (odds - 1) - (1 - p_under - p_push)
        lean = "OVER" if ev_over >= ev_under else "UNDER"
        prob, ev = (p_over, ev_over) if lean == "OVER" else (p_under, ev_under)
        bettable = prob >= min_prob and ev >= min_ev
        if (ln.current_odds_type or "standard") != "standard":
            bettable, _ = False, note.append(f"odds_type:{ln.current_odds_type}")
        if skip_voidable and ln.voidable:
            bettable, _ = False, note.append("voidable")
        if ln.open_line and ln.open_line > 0:
            move = (ln.current_line - ln.open_line) / ln.open_line
            against = (lean == "OVER" and move > 0) or (lean == "UNDER" and move < 0)
            if against and abs(move) > max_move:
                bettable, _ = False, note.append(f"bumped_against:{move:+.0%}")
        if ln.status and ln.status != "pre_game":
            bettable, _ = False, note.append(f"status:{ln.status}")
        rows.append(_row(ln, model.version, computed_at, proj, p_over, p_under, ev_over, ev_under, lean, prob, int(bettable), note, ev))
    out = pd.DataFrame(rows)
    if not out.empty:
        _store(conn, out)
    return out


def _row(ln, version, computed_at, proj, p_over, p_under, ev_over, ev_under, lean, prob, bettable, note, ev=None) -> dict:
    return {
        "book": ln.book, "projection_id": ln.projection_id, "model_version": version, "computed_at": computed_at,
        "sport": ln.sport, "player_name": ln.player_name, "team": ln.team, "opponent": ln.opponent, "stat_type": ln.stat_type,
        "start_time": ln.start_time, "open_line": ln.open_line, "line": ln.current_line, "odds_type": ln.current_odds_type,
        "projection": proj, "p_over": p_over, "p_under": p_under, "ev_over": ev_over, "ev_under": ev_under,
        "lean": lean, "prob": prob, "ev": ev, "bettable": bettable, "notes": ";".join(note), "game_id": ln.game_id,
        "voidable": ln.voidable, "combo": ln.combo,
    }


def _store(conn: sqlite3.Connection, out: pd.DataFrame) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO predictions(book, projection_id, model_version, computed_at, line, projection, p_over, p_under,
               ev_over, ev_under, lean, prob, ev, bettable, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (r.book, r.projection_id, r.model_version, r.computed_at, r.line, _f(r.projection), _f(r.p_over), _f(r.p_under),
             _f(r.ev_over), _f(r.ev_under), r.lean, _f(r.prob), _f(r.ev), int(r.bettable), r.notes)
            for r in out.itertuples(index=False)
        ],
    )
    conn.commit()


def _f(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

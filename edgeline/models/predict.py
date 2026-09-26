"""Price current book lines with trained models: projection, P(over/under/push), EV, bettable flag.

BoardPricer resolves book names to history names, builds one feature row per
(player, map) component, predicts the per-map mean with the stat's model, and
prices single components analytically (NB) or multi-component lines by Gaussian-
copula simulation (multi-map sums, combos). It keeps every priced line's
components so the stacks module can price correlated pairs consistently.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..books.series_format import lol_series_formats, voidable_with_format
from ..config import DEFAULT_MARKET_SHRINK, DEFAULT_MAX_LINE_MOVE, DEFAULT_MIN_EV, DEFAULT_MIN_PROB, UNPROVEN_MARKETS
from ..ev.payouts import leg_decimal_odds
from ..features.build import assemble_prediction_row, current_state
from .banlist import banned_set
from .copula import Component, sum_over_under_push
from .distributions import over_under_push
from .props import PropModel

SUPPORTED_STATS = ("kills", "deaths", "assists", "headshots")
# Call of Duty League map order fixes the game mode (BO7 season rotation); kills differ hugely by mode.
COD_MODE_BY_MAP = {1: "HP", 2: "S&D", 3: "OVL", 4: "HP", 5: "S&D"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# bo3.gg keeps sponsor tags in nicknames ("Eros//Cryptic", "R5 ||SLIGHT", "TP |ogwizard"); books post the bare name.
_TAG_PREFIX = re.compile(r"^[A-Za-z0-9.$]{1,8}\s*(?://|\|\||\|)\s*")
_TAG_SUFFIX = re.compile(r"\s*(?://|\|\||\|)\s*[A-Za-z0-9.$]{1,8}$")


def _core(s: str) -> str:
    """Normalized name with sponsor tags and edge digits ("WUMBO1", "1corim") removed; digits stay when little else is left."""
    s = _TAG_SUFFIX.sub("", _TAG_PREFIX.sub("", (s or "").strip()))
    n = _norm(s)
    stripped = n.strip("0123456789")
    return stripped if len(re.sub(r"[0-9]", "", stripped)) >= 3 else n


class NameResolver:
    """Map book player/team strings onto history names.

    Players: exact, then (when the book team is known) a unique match on that team's roster, then the
    normalized and tag-stripped global indexes, then the prefix before a tag ("NAF-FLY" -> "NAF"), and
    finally a unique roster player whose tag-stripped name contains the book name ("nicx" -> "mesaminicx").
    Teams: exact, normalized, initials, unique prefix.
    """

    def __init__(self, players: list[str], teams: list[str], rosters: dict[str, list[str]] | None = None):
        self.players = {p: p for p in players}
        self.players_norm = {}
        self.players_core = {}
        for p in players:
            self.players_norm.setdefault(_norm(p), p)
            self.players_core.setdefault(_core(p), p)
        self.rosters = {t: list(ps) for t, ps in (rosters or {}).items()}
        self.teams = {t: t for t in teams}
        self.teams_norm = {_norm(t): t for t in teams}
        self.team_initials = {}
        for t in teams:
            words = re.findall(r"[A-Za-z0-9]+", t)
            if len(words) >= 2:
                self.team_initials.setdefault("".join(w[0] for w in words).upper(), t)

    def player(self, name: str, team: str | None = None) -> str | None:
        roster = self.rosters.get(team, []) if team else []
        n, c = _norm(name), _core(name)
        if name in roster:
            return name
        on_roster = [p for p in roster if _norm(p) == n or _core(p) == c]
        if len(on_roster) == 1:
            return on_roster[0]
        if name in self.players:
            return name
        hit = self.players_norm.get(n) or self.players_core.get(c)
        if hit:
            return hit
        base = re.split(r"[-_ ]", name.strip())[0]  # 'NAF-FLY' -> 'NAF'
        if base and base != name:
            hit = self.players_norm.get(_norm(base)) or self.players_core.get(_core(base))
            if hit:
                return hit
        if len(c) >= 4:
            cands = [p for p in roster if len(_core(p)) >= 4 and (c in _core(p) or _core(p) in c)]
            if len(cands) == 1:
                return cands[0]
        return None

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
        cands = [t for k, t in self.teams_norm.items() if k.startswith(n)] if len(n) >= 3 else []
        return cands[0] if len(cands) == 1 else None


ROSTER_DAYS = 180


def rosters(player_state: pd.DataFrame, max_days: float = ROSTER_DAYS) -> dict[str, list[str]]:
    """Current roster per history team: players whose latest game was with that team within `max_days`."""
    out: dict[str, list[str]] = {}
    if "team" not in player_state.columns:
        return out
    days = player_state["p_days_since"] if "p_days_since" in player_state.columns else pd.Series(0.0, index=player_state.index)
    for name, team, d in zip(player_state.index, player_state["team"], days):
        if isinstance(team, str) and team and (pd.isna(d) or d <= max_days):
            out.setdefault(team, []).append(name)
    return out


def resolve_board_teams(lines: pd.DataFrame, player_state: pd.DataFrame, resolver: "NameResolver") -> dict[str, str]:
    """Map each book team code to a history team name by majority vote of the code's players' latest teams."""
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


@dataclass
class PricedLine:
    projection_id: str
    game_id: str | None
    stat: str
    line: float
    components: list[Component]
    projection: float
    p_over: float
    p_under: float
    p_push: float
    p_over_raw: float
    lean: str
    prob: float
    ev: float
    bettable: bool
    notes: list[str] = field(default_factory=list)


class BoardPricer:
    def __init__(self, conn: sqlite3.Connection, sport: str, book: str = "prizepicks", min_prob: float = DEFAULT_MIN_PROB,
                 min_ev: float = DEFAULT_MIN_EV, max_move: float = DEFAULT_MAX_LINE_MOVE, skip_voidable: bool = True,
                 only_upcoming: bool = True, market_shrink: float = DEFAULT_MARKET_SHRINK, include_unproven: bool = False,
                 models: dict | None = None, history_until: str | None = None, use_open_line: bool = False, ignore_status: bool = False,
                 line_ids: list[str] | None = None):
        """Replay mode (models + history_until + use_open_line + ignore_status + line_ids) prices already-settled
        lines at their opening line with models and an as-of state that know nothing after `history_until`."""
        self.conn, self.sport, self.book = conn, sport, book
        self.include_unproven = include_unproven
        self.min_prob, self.min_ev, self.max_move, self.skip_voidable = min_prob, min_ev, max_move, skip_voidable
        self.market_shrink = float(market_shrink)
        self.use_open_line, self.ignore_status = use_open_line, ignore_status
        self.models = models or load_models(sport)
        if not self.models:
            raise FileNotFoundError(f"no trained models for sport={sport}; run `edgeline model train --sport {sport}`")
        if history_until:
            pg = pd.read_sql_query("SELECT * FROM player_games WHERE sport=? AND date < ?", conn, params=(sport, history_until))
        else:
            pg = pd.read_sql_query("SELECT * FROM player_games WHERE sport=?", conn, params=(sport,))
        if pg.empty:
            raise RuntimeError(f"no history rows for sport={sport}")
        self.player_state, self.team_state = current_state(pg, pd.Timestamp(history_until, tz="UTC") if history_until else None)
        self.resolver = NameResolver(list(self.player_state.index), list(self.team_state.index), rosters(self.player_state))
        self.odds = leg_decimal_odds(book)
        self.banned = banned_set(conn, sport)
        lines = pd.read_sql_query("SELECT * FROM lines WHERE book=? AND sport=?", conn, params=(book, sport))
        if line_ids is not None:
            lines = lines[lines["projection_id"].isin(set(line_ids))]
        elif only_upcoming and not lines.empty:
            st = pd.to_datetime(lines["start_time"], utc=True, errors="coerce")
            lines = lines[(st.isna()) | (st > pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1))]
        self.lines = lines.reset_index(drop=True)
        self.team_map = resolve_board_teams(self.lines, self.player_state, self.resolver) if not self.lines.empty else {}
        self.formats = lol_series_formats() if (sport == "lol" and not self.lines.empty) else {}
        self.priced: dict[str, PricedLine] = {}

    def _voidable(self, ln) -> tuple[bool, int | None]:
        """Refine the pull-time voidable flag with the official series format when known."""
        fmt = self.formats.get(frozenset([c for c in (ln.team, ln.opponent) if c])) if self.formats else None
        if fmt is None:
            return bool(ln.voidable), None
        return voidable_with_format(int(ln.map_to), fmt), fmt

    # ---- components -------------------------------------------------------------------------------------------
    def _components(self, ln) -> tuple[PropModel | None, list[Component], list[dict], list[str]]:
        """Resolve a line into (model, components without mu, feature rows, notes)."""
        notes: list[str] = []
        model = self.models.get(ln.stat)
        if model is None:
            return None, [], [], ["no_model_for_stat"]
        names = json.loads(ln.combo_players) if ln.combo and ln.combo_players else [ln.player_name]
        resolved = [self.resolver.player(n, self.team_map.get(ln.team)) for n in names]
        if any(r is None for r in resolved):
            notes.append("player_unmapped:" + ",".join(n for n, r in zip(names, resolved) if r is None))
            return model, [], [], notes
        team_name = self.team_map.get(ln.team) or self.player_state.loc[resolved[0]].get("team")
        opp_name = self.team_map.get(ln.opponent)
        if opp_name is None:
            notes.append("opponent_unmapped")
        team_row = self.team_state.loc[team_name] if team_name in self.team_state.index else None
        opp_row = self.team_state.loc[opp_name] if opp_name in self.team_state.index else None
        maps = list(range(int(ln.map_from), int(ln.map_to) + 1))
        comps, feats = [], []
        for pname in resolved:
            prow = self.player_state.loc[pname]
            p_team = team_name if len(resolved) == 1 else (prow.get("team") or team_name)
            if len(resolved) == 1:
                tenure = prow.get("p_team_games60")
                rest = prow.get("p_days_since")
                if tenure is not None and not pd.isna(tenure):
                    notes.append(f"tenure60:{int(tenure)}")
                if rest is not None and not pd.isna(rest):
                    notes.append(f"rest:{float(rest):.0f}")
            for m in maps:
                role = COD_MODE_BY_MAP.get(m) if self.sport == "cod" else None
                feats.append(assemble_prediction_row(prow, team_row, opp_row, m, 0, None, role))
                comps.append(Component(mu=float("nan"), player=pname, team=p_team, map_index=m))
        return model, comps, feats, notes

    # ---- pricing ----------------------------------------------------------------------------------------------
    def price(self, store: bool = True) -> pd.DataFrame:
        computed_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        prepared = []
        for ln in self.lines.itertuples(index=False):
            model, comps, feats, notes = self._components(ln)
            prepared.append((ln, model, comps, feats, notes))
        # batch mu prediction per model
        by_model: dict[str, list[tuple[int, int]]] = {}
        for i, (_, model, comps, feats, _) in enumerate(prepared):
            if model is not None and feats:
                by_model.setdefault(model.stat, []).extend((i, j) for j in range(len(feats)))
        for stat, idx in by_model.items():
            model = self.models[stat]
            X = pd.DataFrame([prepared[i][3][j] for i, j in idx])
            mus = model.predict_mu(X)
            for (i, j), mu in zip(idx, mus):
                c = prepared[i][2][j]
                prepared[i][2][j] = Component(mu=float(mu), player=c.player, team=c.team, map_index=c.map_index)
        rows = []
        for ln, model, comps, feats, notes in prepared:
            if model is None:
                continue
            if not comps:
                rows.append(self._row(ln, model.version, computed_at, None, notes))
                continue
            line = float(ln.open_line if self.use_open_line and ln.open_line else ln.current_line)
            if self.market_shrink > 0:
                # Market prior: pull each component's mean toward the book's implied per-component line.
                w = self.market_shrink
                share = line / len(comps)
                comps = [Component(mu=(1 - w) * c.mu + w * share, player=c.player, team=c.team, map_index=c.map_index) for c in comps]
                notes.append(f"shrink:{w:g}")
            if len(comps) == 1:
                over, under, push = over_under_push(line, comps[0].mu, model.r)
            else:
                over, under, push = sum_over_under_push(line, comps, model.r, model.rho_self, model.rho_team, model.rho_opp)
            over_c = float(model.calibrate(over))
            under_c = max(0.0, 1.0 - over_c - push)
            odds_o = float(ln.odds_over) if getattr(ln, "odds_over", None) else self.odds  # per-pick odds (Sleeper) else the book's leg odds
            odds_u = float(ln.odds_under) if getattr(ln, "odds_under", None) else self.odds
            ev_over = over_c * (odds_o - 1) - (1 - over_c - push)
            ev_under = under_c * (odds_u - 1) - (1 - under_c - push)
            lean = "OVER" if ev_over >= ev_under else "UNDER"
            prob, ev = (over_c, ev_over) if lean == "OVER" else (under_c, ev_under)
            bettable = prob >= self.min_prob and ev >= self.min_ev
            if not self.include_unproven and market_is_unproven(self.sport, ln.stat):
                bettable = False
                notes.append("market_unproven")
            if (ln.current_odds_type or "standard") != "standard":
                bettable = False
                notes.append(f"odds_type:{ln.current_odds_type}")
            voidable, fmt = self._voidable(ln)
            if fmt:
                notes.append(f"bo{fmt}")
            if self.skip_voidable and voidable:
                bettable = False
                notes.append("voidable")
            if ln.open_line and ln.open_line > 0 and not self.use_open_line:
                move = (ln.current_line - ln.open_line) / ln.open_line
                against = (lean == "OVER" and move > 0) or (lean == "UNDER" and move < 0)
                if against and abs(move) > self.max_move:
                    bettable = False
                    notes.append(f"bumped_against:{move:+.0%}")
            if ln.status and ln.status != "pre_game" and not self.ignore_status:
                bettable = False
                notes.append(f"status:{ln.status}")
            if any(c.player in self.banned for c in comps):
                bettable = False
                notes.append("banned_player")
            pl = PricedLine(ln.projection_id, ln.game_id, ln.stat, line, comps, float(sum(c.mu for c in comps)), over_c, under_c,
                            float(push), float(over), lean, float(prob), float(ev), bool(bettable), notes)
            self.priced[ln.projection_id] = pl
            rows.append(self._row(ln, model.version, computed_at, pl, notes, ev_over, ev_under))
        out = pd.DataFrame(rows)
        if store and not out.empty:
            _store(self.conn, out)
        return out

    def _row(self, ln, version, computed_at, pl: PricedLine | None, notes, ev_over=None, ev_under=None) -> dict:
        return {
            "book": ln.book, "projection_id": ln.projection_id, "model_version": version, "computed_at": computed_at,
            "sport": ln.sport, "player_name": ln.player_name, "team": ln.team, "opponent": ln.opponent, "stat_type": ln.stat_type,
            "start_time": ln.start_time, "open_line": ln.open_line, "line": ln.current_line, "odds_type": ln.current_odds_type,
            "projection": None if pl is None else pl.projection, "p_over": None if pl is None else pl.p_over,
            "p_under": None if pl is None else pl.p_under, "p_push": None if pl is None else pl.p_push,
            "ev_over": ev_over, "ev_under": ev_under, "lean": None if pl is None else pl.lean, "prob": None if pl is None else pl.prob,
            "ev": None if pl is None else pl.ev, "bettable": 0 if pl is None else int(pl.bettable), "notes": ";".join(notes),
            "game_id": ln.game_id, "voidable": ln.voidable, "combo": ln.combo,
        }


def market_is_unproven(sport: str, stat: str | None) -> bool:
    """Markets the walk-forward backtest and the captured-line record do not support at the 60% threshold."""
    return (sport, stat) in UNPROVEN_MARKETS


def price_board(conn: sqlite3.Connection, sport: str, book: str = "prizepicks", min_prob: float = DEFAULT_MIN_PROB,
                min_ev: float = DEFAULT_MIN_EV, max_move: float = DEFAULT_MAX_LINE_MOVE, skip_voidable: bool = True,
                only_upcoming: bool = True, market_shrink: float = DEFAULT_MARKET_SHRINK, include_unproven: bool = False) -> pd.DataFrame:
    return BoardPricer(conn, sport, book, min_prob, min_ev, max_move, skip_voidable, only_upcoming, market_shrink, include_unproven).price()


def _store(conn: sqlite3.Connection, out: pd.DataFrame) -> None:
    # A re-price supersedes every earlier model version's row for the same projection.
    conn.executemany("DELETE FROM predictions WHERE book=? AND projection_id=?", [(r.book, r.projection_id) for r in out.itertuples(index=False)])
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

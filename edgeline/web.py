"""Feeds for a website in oddsfloor.com's shapes: the board's `opportunities` list, the best slips, and the tracker.

`edgeline web` writes data/web/v1/esports-ev.json (one opportunity per upcoming standard line with a lean; the board
filters by `tier`), esports-slips.json (the best slip at every price, with stakes) and esports-record.json (the
real-line record with match-cluster intervals). Field names follow the board's own JavaScript (bookKey, market,
point, outcome, participant, decimalOdds, fairProb, ev, eventId, commenceTime, homeTeam, awayTeam, sportKey, link,
kelly, tier) so the existing screener can render them; extra fields carry what the model adds.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

import pandas as pd

from .config import DATA_DIR
from .ev.payouts import leg_decimal_odds
from .grading.results import results_frame
from .grading.roi import MIN_CLUSTERS, cluster_ci, match_key, wilson
from .slips.best import best_per_price, best_slips
from .slips.horizon import horizon_end, still_listed, within_horizon

WEB_DIR = DATA_DIR / "web" / "v1"
SPORT_KEYS = {"cs2": "esports_cs2", "lol": "esports_lol", "dota": "esports_dota2", "val": "esports_valorant", "cod": "esports_cod"}


def _market_key(stat_type: str) -> str:
    return stat_type.lower().replace(" (combo)", "_combo").replace("-", "_").replace(" ", "_")


def opportunities(conn: sqlite3.Connection, book: str = "prizepicks", days: int | None = None) -> list[dict]:
    q = """
    SELECT p.projection_id, p.lean, p.prob, p.ev, p.projection, p.bettable, p.notes, p.computed_at, p.model_version,
           l.sport, l.player_name, l.team, l.opponent, l.game_id, l.stat_type, l.stat, l.map_from, l.map_to, l.combo,
           l.start_time, l.current_line, l.open_line, l.current_odds_type, l.last_seen_at
    FROM predictions p JOIN lines l ON l.book=p.book AND l.projection_id=p.projection_id
    WHERE p.book=? AND p.lean IS NOT NULL AND COALESCE(l.current_odds_type, 'standard')='standard'
      AND l.start_time > strftime('%Y-%m-%dT%H:%M:%SZ','now')
    """
    df = pd.read_sql_query(q, conn, params=(book,))
    if df.empty:
        return []
    df = df.sort_values("computed_at").groupby("projection_id", as_index=False).tail(1)
    df = df[still_listed(conn, df, book)]
    inside = within_horizon(df["start_time"], days)
    odds = leg_decimal_odds(book)
    out = []
    for r, horizon in zip(df.itertuples(index=False), inside):
        notes = set((r.notes or "").split(";"))
        gated = "market_unproven" in notes
        voidable = "voidable" in notes
        tier = "bettable" if r.bettable else ("gated" if gated else ("voidable" if voidable else "lean"))
        side = "u" if r.lean == "UNDER" else "o"
        out.append({
            "id": f"{book}:{r.projection_id}", "bookKey": book, "sportKey": SPORT_KEYS.get(r.sport, f"esports_{r.sport}"),
            "eventId": r.game_id, "commenceTime": pd.Timestamp(r.start_time).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ") if pd.Timestamp(r.start_time).tzinfo else r.start_time,
            "homeTeam": r.team, "awayTeam": r.opponent, "market": _market_key(r.stat_type), "marketLabel": r.stat_type,
            "participant": r.player_name, "outcome": "Under" if r.lean == "UNDER" else "Over", "point": float(r.current_line),
            "openPoint": float(r.open_line), "decimalOdds": round(odds, 4), "fairProb": round(float(r.prob), 4), "ev": round(float(r.ev), 4),
            "projection": None if r.projection is None or pd.isna(r.projection) else round(float(r.projection), 2),
            "tier": tier, "inHorizon": bool(horizon), "voidable": voidable, "gated": gated, "combo": bool(r.combo),
            "link": f"https://app.prizepicks.com/?projections={r.projection_id}-{side}-{float(r.current_line):g}" if book == "prizepicks" else None,
            "linkKind": "deeplink", "derivedFrom": r.model_version, "quoteAgeMs": int(max(0.0, (pd.Timestamp.now(tz="UTC") - pd.Timestamp(r.last_seen_at)).total_seconds() * 1000)),
        })
    out.sort(key=lambda o: (-int(o["tier"] == "bettable"), -o["fairProb"]))
    return out


def slips(conn: sqlite3.Connection, book: str = "prizepicks", bankroll: float = 2000.0, kelly: float = 0.25, cap: float = 0.005,
          days: int | None = None) -> list[dict]:
    cands = best_slips(conn, book, days=days)
    rows = []
    for c in best_per_price(cands):
        stake = min(c.kelly * kelly, cap) * bankroll if c.growth > 0 else 0.0
        rows.append({"price": f"{c.slip_type} {c.size}", "slipType": c.slip_type, "size": c.size, "topPayout": round(c.net[-1] + 1, 2), "kind": c.kind,
                     "ev": round(c.ev, 4), "pTop": round(c.p_top, 4), "pPaid": round(c.p_paid, 4), "growth": round(c.growth, 5), "kelly": round(c.kelly, 4),
                     "stake": round(stake, 2), "legs": c.legs, "link": c.link, "label": c.label})
    return rows


def record(conn: sqlite3.Connection, book: str = "prizepicks") -> dict:
    base = {"breakEvenLeg": round(1.0 / leg_decimal_odds(book), 4), "targetLeg": 0.60}
    df = results_frame(conn, book)
    if df.empty or "win_open" not in df.columns:
        return {**base, "slices": []}
    df = df.dropna(subset=["win_open"])
    if df.empty:
        return {**base, "slices": []}

    def gauge(label: str, g: pd.DataFrame) -> dict:
        n, w = len(g), int(g["win_open"].sum())
        p, lo, hi = wilson(w, n) if n else (float("nan"),) * 3
        m, clo, chi = cluster_ci(g["win_open"], match_key(g)) if n else (0, float("nan"), float("nan"))
        return {"slice": label, "n": n, "wins": w, "losses": n - w, "hitRate": round(p, 4), "ciLow": round(lo, 4), "ciHigh": round(hi, 4),
                "matches": int(m), "clusterCiLow": None if m < MIN_CLUSTERS else round(clo, 4), "clusterCiHigh": None if m < MIN_CLUSTERS else round(chi, 4)}

    slices = [gauge("all leans", df), gauge("bettable", df[df["bettable"] == 1])]
    for sp, g in df.groupby("sport"):
        slices.append(gauge(f"{sp} leans", g))
        slices.append(gauge(f"{sp} bettable", g[g["bettable"] == 1]))
    return {**base, "slices": slices}


def write_feeds(conn: sqlite3.Connection, book: str = "prizepicks", bankroll: float = 2000.0, out_dir: Path = WEB_DIR, days: int | None = None) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    ops = opportunities(conn, book, days)
    sl = slips(conn, book, bankroll, days=days)
    rec = record(conn, book)
    horizon = horizon_end(days).isoformat()
    (out_dir / "esports-ev.json").write_text(json.dumps({"generatedAt": now, "demo": False, "book": book, "horizonEnd": horizon, "count": len(ops), "opportunities": ops}, indent=0))
    (out_dir / "esports-slips.json").write_text(json.dumps({"generatedAt": now, "book": book, "bankroll": bankroll, "horizonEnd": horizon, "count": len(sl), "slips": sl}, indent=0))
    (out_dir / "esports-record.json").write_text(json.dumps({"generatedAt": now, "book": book, **rec}, indent=0))
    return {"opportunities": len(ops), "slips": len(sl), "recordSlices": len(rec.get("slices", []))}

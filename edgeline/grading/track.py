"""Day-on-day tracking of one market on real lines, so a gated market's return (or a live market's decline) is
visible per match day instead of buried in the running total.

Each row is a match day (UTC start date): our leans' record, the number of matches, the hit rate, and the
blind rates a bettor who took every under (or every over) would have had on the same lines. The selection
value is our lean hit rate minus the blind rate of the same side, which is what the model adds over the
market's own tilt. The gate lifts on a pre-stated rule, not on a good day.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from ..config import UNPROVEN_MARKETS
from .results import results_frame
from .roi import match_key, wilson

LIFT_MIN_LINES = 150
LIFT_MIN_MATCHES = 10
LIFT_MIN_HIT = 0.58
LIFT_MIN_VALUE = 0.03


def market_frame(conn: sqlite3.Connection, sport: str, stat: str, book: str = "prizepicks") -> pd.DataFrame:
    df = results_frame(conn, book)
    if df.empty:
        return df
    df = df[(df["sport"] == sport) & df["result_open"].isin(["over", "under"])].copy()
    df["stat"] = df["stat_type"].str.lower().str.replace(r"^map[s]? ?[0-9-]+ ", "", regex=True).str.replace(" (combo)", "", regex=False)
    df = df[df["stat"] == stat].copy()
    if df.empty:
        return df
    df["day"] = pd.to_datetime(df["start_time"], utc=True, format="ISO8601").dt.strftime("%Y-%m-%d")
    df["match"] = match_key(df)
    df["lean_side"] = df["lean"].str.lower()
    df["went_under"] = (df["result_open"] == "under").astype(float)
    return df


def _row(g: pd.DataFrame) -> dict:
    n, w = len(g), int(g["win_open"].sum())
    under = g[g["lean_side"] == "under"]; over = g[g["lean_side"] == "over"]
    blind_under = g["went_under"].mean()
    hit_under = under["win_open"].mean() if len(under) else float("nan")
    hit_over = over["win_open"].mean() if len(over) else float("nan")
    # selection value: how far our leans beat a blind bet on the same side
    value = 0.0
    if len(under):
        value += (hit_under - blind_under) * len(under)
    if len(over):
        value += (hit_over - (1 - blind_under)) * len(over)
    return {"lines": n, "record": f"{w}-{n - w}", "hit": w / n if n else float("nan"), "matches": g["match"].nunique(),
            "under leans": len(under), "under hit": hit_under, "over leans": len(over), "over hit": hit_over,
            "blind under": blind_under, "selection value": value / n if n else float("nan")}


def daily(df: pd.DataFrame, days: int = 14) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for day, g in df.groupby("day"):
        rows.append({"day": day, **_row(g)})
    out = pd.DataFrame(rows).sort_values("day").tail(days)
    return out.reset_index(drop=True)


def trailing(df: pd.DataFrame, matches: int = LIFT_MIN_MATCHES) -> dict:
    """The most recent `matches` matches (by start day then match key), with the lift-rule verdict."""
    if df.empty:
        return {"lines": 0}
    order = df.groupby("match")["start_time"].min().sort_values()
    recent = order.index[-matches:]
    g = df[df["match"].isin(recent)]
    r = _row(g)
    n, w = r["lines"], int(g["win_open"].sum())
    _, lo, hi = wilson(w, n)
    r.update({"ci_low": lo, "ci_high": hi})
    r["lift_rule"] = bool(n >= LIFT_MIN_LINES and r["matches"] >= LIFT_MIN_MATCHES and r["hit"] >= LIFT_MIN_HIT and r["selection value"] >= LIFT_MIN_VALUE)
    return r


def report(conn: sqlite3.Connection, sport: str, stat: str, book: str = "prizepicks", days: int = 14) -> str:
    df = market_frame(conn, sport, stat, book)
    gated = (sport, stat) in UNPROVEN_MARKETS
    head = f"{sport} {stat} on {book}: {'GATED (priced and graded, not flagged bettable)' if gated else 'open'}"
    if df.empty:
        return head + "\nno settled leans yet"
    d = daily(df, days).copy()
    for c in ("hit", "under hit", "over hit", "blind under", "selection value"):
        d[c] = (d[c] * 100).round(1)
    t = trailing(df)
    lines = [head, d.to_string(index=False),
             f"trailing {t['matches']} matches: {t['record']} ({t['hit']:.1%}, 95% CI {t['ci_low']:.1%} to {t['ci_high']:.1%}), "
             f"selection value {t['selection value']:+.1%} over the blind side",
             f"lift rule ({LIFT_MIN_LINES}+ lines over {LIFT_MIN_MATCHES}+ matches, hit >= {LIFT_MIN_HIT:.0%}, selection value >= {LIFT_MIN_VALUE:+.0%}): "
             + ("MET" if t["lift_rule"] else "not met"),
             "hit/under hit/over hit/blind under/selection value in %; blind under = share of all lines that settled under;",
             "selection value = our leans' hit rate minus the blind rate of the same side, i.e. what the model adds over the market's tilt."]
    return "\n".join(lines)

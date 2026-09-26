"""Replay: price already-settled captured lines at their opening line with models that know nothing after a cutoff.

The forward test on captured lines takes weeks to accumulate; a replay answers "would today's model have beaten
the book on the lines already captured" in an hour. Models are trained only on games before `cutoff`, the as-of
player and team state is built from the same games, and every settled line that started after the cutoff is
priced at its opening line and graded against the stored result. It is out of sample on real lines, with the
one caveat that the code and feature choices were made knowing those days' aggregate results.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import joblib
import pandas as pd

from ..config import DEFAULT_MARKET_SHRINK
from ..features.build import FEATURE_COLUMNS, build_training_frame
from ..grading.roi import wilson
from .predict import SUPPORTED_STATS, BoardPricer
from .props import train


CACHE_DIR = Path("data/backtests/cache")


def _cutoff_model(frame: pd.DataFrame, sport: str, stat: str, cutoff_iso: str):
    """train() cached per (sport, stat, cutoff, rows) so pricing variants (shrink, thresholds) re-use the same models."""
    key = f"replay_{sport}_{stat}_{cutoff_iso[:10]}_{len(frame)}_{abs(hash(tuple(FEATURE_COLUMNS))) % 10**8}"
    path = CACHE_DIR / f"{key}.joblib"
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            pass
    model = train(frame, sport, stat)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model


def replay(conn: sqlite3.Connection, sport: str, cutoff: str, book: str = "prizepicks", shrink: float = DEFAULT_MARKET_SHRINK, progress=None) -> pd.DataFrame:
    cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
    cutoff_iso = cutoff_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = pd.read_sql_query(
        "SELECT l.projection_id, l.stat, l.stat_type, l.player_name, l.map_from, l.map_to, l.start_time, l.open_line, g.result_open, g.actual "
        "FROM lines l JOIN grades g ON g.book=l.book AND g.projection_id=l.projection_id "
        "WHERE l.book=? AND l.sport=? AND g.result_open IN ('over','under')", conn, params=(book, sport))
    lines = lines[pd.to_datetime(lines["start_time"], utc=True, errors="coerce") >= cutoff_ts]
    stats = sorted(set(lines["stat"].dropna()) & set(SUPPORTED_STATS))
    if lines.empty or not stats:
        return pd.DataFrame()
    pg = pd.read_sql_query("SELECT * FROM player_games WHERE sport=? AND date < ?", conn, params=(sport, cutoff_iso))
    frame = build_training_frame(pg)
    models = {}
    for st in stats:
        if progress:
            progress(f"{sport}/{st}: training on {len(frame):,} rows before {cutoff_iso}")
        models[st] = _cutoff_model(frame, sport, st, cutoff_iso)
    pricer = BoardPricer(conn, sport, book, only_upcoming=False, models=models, history_until=cutoff_iso, use_open_line=True,
                         ignore_status=True, line_ids=lines["projection_id"].tolist(), market_shrink=shrink)
    out = pricer.price(store=False)
    if out.empty:
        return out
    merged = out.merge(lines[["projection_id", "stat", "map_from", "map_to", "result_open", "actual"]], on="projection_id", suffixes=("", "_line"))
    merged = merged[merged["lean"].notna()].copy()
    merged["win"] = (merged["lean"].str.lower() == merged["result_open"]).astype(float)
    return merged


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame(rows)
    groups = [("all leans", df), ("bettable", df[df["bettable"] == 1])]
    for st, g in df.groupby("stat"):
        groups.append((f"{st} leans", g))
        groups.append((f"{st} bettable", g[g["bettable"] == 1]))
    for name, g in groups:
        n, w = int(len(g)), int(g["win"].sum())
        p, lo, hi = wilson(w, n) if n else (float("nan"),) * 3
        rows.append({"slice": name, "n": n, "wins": w, "losses": n - w, "hit_rate": p, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)

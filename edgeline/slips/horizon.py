"""Start-time horizon for anything proposed as a bet: today or tomorrow in the bettor's local day."""
from __future__ import annotations

import pandas as pd

from ..config import LOCAL_TZ, SLIP_HORIZON_DAYS


def horizon_end(days: int | None = None, now: pd.Timestamp | None = None, tz: str = LOCAL_TZ) -> pd.Timestamp:
    """Midnight (local) at the end of `days` days from today: days=0 is today only, days=1 today and tomorrow."""
    days = SLIP_HORIZON_DAYS if days is None else days
    now = (now or pd.Timestamp.now(tz="UTC")).tz_convert(tz)
    return (now.normalize() + pd.Timedelta(days=days + 1)).tz_convert("UTC")


def within_horizon(start_time: pd.Series, days: int | None = None, now: pd.Timestamp | None = None, tz: str = LOCAL_TZ) -> pd.Series:
    """Boolean mask: the line's start is before the horizon end (lines with no start time are kept)."""
    st = pd.to_datetime(start_time, utc=True, errors="coerce")
    return st.isna() | (st < horizon_end(days, now, tz))


FRESH_MINUTES = 30


def still_listed(conn, df: pd.DataFrame, book: str, minutes: int = FRESH_MINUTES) -> pd.Series:
    """Boolean mask: the line was seen within `minutes` of the book's latest pull (lines the book has pulled from the
    board keep their last_seen_at, so they fall out of the window and are never proposed)."""
    latest = conn.execute("SELECT MAX(last_seen_at) FROM lines WHERE book=?", (book,)).fetchone()[0]
    if not latest or "last_seen_at" not in df.columns:
        return pd.Series(True, index=df.index)
    seen = pd.to_datetime(df["last_seen_at"], utc=True, errors="coerce")
    return seen >= pd.Timestamp(latest).tz_convert("UTC") - pd.Timedelta(minutes=minutes) if pd.Timestamp(latest).tzinfo else seen >= pd.Timestamp(latest, tz="UTC") - pd.Timedelta(minutes=minutes)

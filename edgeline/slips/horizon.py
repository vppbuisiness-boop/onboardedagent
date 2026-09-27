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

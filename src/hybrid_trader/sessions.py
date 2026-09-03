from __future__ import annotations

import exchange_calendars as xcals
import pandas as pd

from .data import validate_ohlcv


def regular_nyse_session(bars: pd.DataFrame) -> pd.DataFrame:
    validated = validate_ohlcv(bars)
    if validated.empty:
        return validated
    calendar = xcals.get_calendar("XNYS")
    schedule = calendar.schedule.loc[
        validated.index.min().date().isoformat() : validated.index.max().date().isoformat()
    ]
    session_dates = validated.index.normalize().tz_localize(None)
    opens = pd.DatetimeIndex(schedule["open"].reindex(session_dates))
    closes = pd.DatetimeIndex(schedule["close"].reindex(session_dates))
    mask = (validated.index >= opens) & (validated.index < closes)
    return validated.loc[mask]

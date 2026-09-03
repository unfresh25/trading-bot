import pandas as pd

from hybrid_trader.sessions import regular_nyse_session


def test_nyse_filter_removes_premarket_and_keeps_regular_bars():
    index = pd.DatetimeIndex(
        [
            "2025-07-02T13:15:00Z",
            "2025-07-02T13:30:00Z",
            "2025-07-02T19:45:00Z",
            "2025-07-02T20:00:00Z",
        ]
    )
    bars = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=index,
    )

    filtered = regular_nyse_session(bars)

    assert filtered.index.tolist() == [index[1], index[2]]


def test_nyse_filter_honors_early_close():
    index = pd.DatetimeIndex(["2025-07-03T16:45:00Z", "2025-07-03T17:00:00Z"])
    bars = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=index,
    )

    filtered = regular_nyse_session(bars)

    assert filtered.index.tolist() == [index[0]]

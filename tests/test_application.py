import pandas as pd

from hybrid_trader.application import prepare_market_bars


def test_prepare_spy_uses_nyse_regular_session_while_crypto_remains_24_7():
    index = pd.DatetimeIndex(["2025-07-02T13:15:00Z", "2025-07-02T13:30:00Z"])
    bars = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=index,
    )

    spy = prepare_market_bars("SPY", bars)
    bitcoin = prepare_market_bars("BTC/USD", bars)

    assert spy.index.tolist() == [index[1]]
    assert bitcoin.index.tolist() == index.tolist()

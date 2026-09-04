import pandas as pd
import pytest

from hybrid_trader.config import AppConfig
from hybrid_trader.engine import _execute_trade, _resolve_exit


def _bars(lows: list[float], opens: list[float]) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=len(lows), freq="15min", tz="UTC")
    highs = [max(o, 1.0) + 5 for o in opens]
    closes = opens
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1000.0},
        index=index,
    )


def test_intrabar_stop_closes_at_stop_price_before_time_exit():
    # entry at open[1]=100, stop_distance = 2*ATR. A later bar's low pierces the stop.
    opens = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    lows = [100.0, 100.0, 100.0, 90.0, 100.0, 100.0]  # bar 3 low=90 pierces stop
    bars = _bars(lows, opens)
    entry_position, exit_position = 1, 5
    stop_price = 100.0 - 2.0 * 3.0  # atr=3, multiple=2 -> stop at 94

    exit_price, resolved, on_stop = _resolve_exit(bars, entry_position, exit_position, stop_price)

    assert on_stop is True
    assert exit_price == pytest.approx(94.0)
    assert resolved == 3  # closed on the breaching bar, not the planned exit


def test_time_exit_stands_when_stop_never_pierced():
    opens = [100.0, 100.0, 101.0, 102.0, 103.0, 104.0]
    lows = [100.0, 99.5, 100.5, 101.5, 102.5, 103.5]  # never below stop 94
    bars = _bars(lows, opens)
    exit_price, resolved, on_stop = _resolve_exit(bars, 1, 5, 94.0)

    assert on_stop is False
    assert exit_price == pytest.approx(104.0)  # open[5]
    assert resolved == 5


def test_execute_trade_bounds_loss_to_sized_risk_on_stop():
    config = AppConfig()
    opens = [100.0, 100.0, 100.0, 80.0, 100.0, 100.0]
    lows = [100.0, 100.0, 100.0, 70.0, 100.0, 100.0]  # deep gap pierces stop on bar 3
    bars = _bars(lows, opens)
    row = pd.Series({"fold": 0, "regime": "trend_low_vol"})
    equity, atr = 500.0, 3.0

    trade = _execute_trade(bars, bars.index[0], row, 0, equity, atr, config)

    assert trade is not None
    assert trade["exited_on_stop"] is True
    # Loss is bounded near the sized risk (0.5% of equity), not the full -20% gap.
    sized_risk = equity * config.risk.risk_fraction
    assert -trade["net_pnl"] <= sized_risk * 1.2  # small tolerance for costs

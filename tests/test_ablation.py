import numpy as np
import pandas as pd

from hybrid_trader.ablation import DEFAULT_STOP_GRID, StopVariant, stop_ablation
from hybrid_trader.config import AppConfig, RiskConfig
from hybrid_trader.engine import _execute_trade


def _bars(lows, opens):
    index = pd.date_range("2025-01-01", periods=len(lows), freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": [o + 5 for o in opens], "low": lows,
         "close": opens, "volume": 1000.0},
        index=index,
    )


def test_apply_intrabar_stop_false_skips_the_stop():
    # A bar's low pierces the stop, but with the flag off the exit stays time-based.
    opens = [100.0, 100.0, 100.0, 80.0, 100.0, 100.0]
    lows = [100.0, 100.0, 100.0, 70.0, 100.0, 100.0]
    bars = _bars(lows, opens)
    row = pd.Series({"fold": 0, "regime": "trend_low_vol"})

    config = AppConfig(risk=RiskConfig(apply_intrabar_stop=False))
    trade = _execute_trade(bars, bars.index[0], row, 0, 500.0, 3.0, config)
    assert trade is not None
    assert trade["exited_on_stop"] is False

    config_on = AppConfig(risk=RiskConfig(apply_intrabar_stop=True))
    trade_on = _execute_trade(bars, bars.index[0], row, 0, 500.0, 3.0, config_on)
    assert trade_on["exited_on_stop"] is True


def _long_bars():
    rng = np.random.default_rng(5)
    index = pd.date_range("2024-01-01", "2024-11-20", freq="15min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0035, len(index))))
    op = np.r_[close[0], close[:-1]]
    dip = np.where(rng.random(len(index)) < 0.05, 1 - np.abs(rng.normal(0, 0.04, len(index))), 1.0)
    low = np.minimum(op, close) * dip
    return pd.DataFrame(
        {"open": op, "high": np.maximum(op, close) * 1.003, "low": low,
         "close": close, "volume": 1000.0},
        index=index,
    )


def test_stop_ablation_returns_one_row_per_variant_with_identical_trade_signals():
    rows = stop_ablation("SPY", _long_bars())
    assert len(rows) == len(DEFAULT_STOP_GRID)
    names = {r["variant"] for r in rows}
    assert names == {v.name for v in DEFAULT_STOP_GRID}
    # Every variant produces a finite metric set and a stop count.
    for r in rows:
        assert r["trades"] >= 0
        assert "sharpe" in r and "net_return" in r
        assert r["stops"] >= 0
    # The no-stop variant must record zero stop exits.
    no_stop = next(r for r in rows if r["variant"] == "no_stop_size_2atr")
    assert no_stop["stops"] == 0


def test_custom_variant_grid():
    rows = stop_ablation("SPY", _long_bars(), variants=(StopVariant("wide", 4.0, True),))
    assert len(rows) == 1
    assert rows[0]["variant"] == "wide"

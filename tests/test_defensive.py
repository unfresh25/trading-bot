import numpy as np
import pandas as pd

from hybrid_trader.config import CostConfig
from hybrid_trader.defensive import defensive_overlay


def _bars(n=500, seed=1):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-01", periods=n, freq="15min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.004, n)))
    return pd.DataFrame({"close": close}, index=index)


def test_infinite_threshold_is_plain_buy_and_hold():
    bars = _bars()
    signal = pd.Series(np.random.default_rng(0).random(len(bars)) * 10, index=bars.index)
    overlay = defensive_overlay(bars, signal, float("inf"), CostConfig())
    # Never steps out -> fully invested, zero switches.
    assert overlay.exposure_mean == 1.0
    assert overlay.switches == 0


def test_overlay_is_causal():
    # Exposure at t uses the signal at t-1; mutating the last bar's signal cannot change
    # any earlier exposure or the equity path up to the penultimate bar.
    bars = _bars()
    signal = pd.Series(0.0, index=bars.index)
    signal.iloc[100] = 10.0  # a single risk spike
    base = defensive_overlay(bars, signal, 5.0, CostConfig(), cooldown_bars=10)
    changed = signal.copy()
    changed.iloc[-1] = 99.0  # mutate the future
    mutated = defensive_overlay(bars, changed, 5.0, CostConfig(), cooldown_bars=10)
    # A future-only change leaves the risk/return realised before it unchanged.
    assert base.max_drawdown == mutated.max_drawdown


def test_elevated_signal_reduces_exposure_with_hysteresis():
    bars = _bars()
    signal = pd.Series(0.0, index=bars.index)
    signal.iloc[200] = 10.0  # one spike
    overlay = defensive_overlay(bars, signal, 5.0, CostConfig(), cooldown_bars=20)
    # One spike -> flat for the cooldown window, then re-enters: exactly two switches.
    assert overlay.exposure_mean < 1.0
    assert overlay.switches == 2

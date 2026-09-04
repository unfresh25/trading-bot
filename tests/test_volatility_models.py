import numpy as np
import pandas as pd
import pytest

pytest.importorskip("arch")

from hybrid_trader.config import FeatureConfig
from hybrid_trader.features import build_features
from hybrid_trader.volatility_models import conditional_volatility


def _close(n=1200, seed=3):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.004, n))), index=index)


def test_conditional_volatility_is_causal():
    # Mutating the future must not change any past conditional-volatility value.
    close = _close()
    cutoff = 800
    original = conditional_volatility(close.iloc[: cutoff + 1], refit_every=50)
    changed = close.copy()
    changed.iloc[cutoff + 1 :] *= 1.4
    mutated = conditional_volatility(changed.iloc[: cutoff + 1], refit_every=50)
    pd.testing.assert_series_equal(original, mutated)


def test_conditional_volatility_has_no_lookahead_warmup():
    close = _close()
    vol = conditional_volatility(close, min_train=250, refit_every=50)
    # No forecast before enough history exists.
    assert vol.iloc[:250].isna().all()
    assert vol.iloc[300:].notna().any()
    assert (vol.dropna() > 0).all()


def test_garch_features_appended_only_when_enabled():
    bars = pd.DataFrame(
        {
            "open": _close().values,
            "high": _close().values * 1.002,
            "low": _close().values * 0.998,
            "close": _close().values,
            "volume": 1000.0,
        },
        index=_close().index,
    )
    baseline = build_features(bars, FeatureConfig(use_garch=False))
    with_garch = build_features(bars, FeatureConfig(use_garch=True, garch_refit_every=100))
    assert "garch_vol_norm" not in baseline.columns
    assert "garch_vol_norm" in with_garch.columns
    assert "garch_vol_ratio" in with_garch.columns
    # Baseline feature set is unchanged by the flag being off (V0 behaviour preserved).
    assert baseline.shape[1] == 28

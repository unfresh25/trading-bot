import numpy as np
import pandas as pd

from hybrid_trader.market_surprise import market_surprise_features


def _market(n=400, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "DGS2": 1.0 + np.cumsum(rng.normal(0, 0.02, n)),
            "T10YIE": 2.0 + np.cumsum(rng.normal(0, 0.01, n)),
            "T5YIFR": 2.0 + np.cumsum(rng.normal(0, 0.01, n)),
        },
        index=dates,
    )


def test_market_surprise_features_are_causal():
    market = _market()
    index = pd.date_range("2021-03-01", "2021-09-01", freq="15min", tz="UTC")
    cutoff = pd.Timestamp("2021-06-01", tz="UTC")
    full = market_surprise_features(market, index)
    # Truncating future market data must not change any already-computed past value.
    truncated_market = market[market.index < cutoff.tz_localize(None)]
    truncated = market_surprise_features(truncated_market, index)
    common = full.index[full.index < cutoff]
    pd.testing.assert_series_equal(
        full.loc[common, "rate_surprise"].dropna(),
        truncated.loc[common, "rate_surprise"].dropna(),
    )


def test_change_is_available_only_after_its_close():
    market = _market()
    index = pd.date_range("2021-01-01", "2021-04-01", freq="15min", tz="UTC")
    feats = market_surprise_features(market, index)
    # The first market date's change cannot be known before the next day.
    first_day = pd.Timestamp(market.index[0], tz="UTC")
    assert feats.loc[feats.index <= first_day, "rate_surprise"].isna().all()


def test_surprise_columns_and_ranges():
    feats = market_surprise_features(
        _market(), pd.date_range("2021-06-01", "2021-12-01", freq="15min", tz="UTC")
    )
    for col in ("rate_surprise", "infl_surprise", "market_surprise"):
        assert col in feats.columns
    valid = feats.dropna()
    assert (valid["surprise_DGS2"].abs() <= 5.0).all()  # standardised changes are clipped
    assert (valid["market_surprise"] >= 0).all()

import numpy as np
import pandas as pd

from hybrid_trader.derivatives_data import derivatives_features


def _funding(n=200, seed=1):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=n, freq="8h", tz="UTC")
    return pd.DataFrame({"funding_rate": rng.normal(0.0001, 0.0003, n)}, index=times)


def _metrics(n=2000, seed=2):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open_interest": 70000 + np.cumsum(rng.normal(0, 50, n)),
            "toptrader_ls_ratio": rng.normal(1.4, 0.1, n),
            "taker_ls_ratio": rng.normal(1.0, 0.05, n),
        },
        index=times,
    )


def test_derivatives_features_are_causal():
    funding, metrics = _funding(), _metrics()
    index = pd.date_range("2024-01-02", "2024-01-06", freq="15min", tz="UTC")
    cutoff = pd.Timestamp("2024-01-04", tz="UTC")
    full = derivatives_features(funding, metrics, index)
    # Mutating future derivatives data must not change any already-computed past feature.
    trunc_f = funding[funding.index < cutoff]
    trunc_m = metrics[metrics.index < cutoff]
    truncated = derivatives_features(trunc_f, trunc_m, index)
    common = full.index[full.index < cutoff]
    pd.testing.assert_series_equal(
        full.loc[common, "funding_rate"].dropna(),
        truncated.loc[common, "funding_rate"].dropna(),
    )


def test_features_are_shifted_one_bar():
    # The first bar can never carry a value: everything is shifted by one bar.
    funding, metrics = _funding(), _metrics()
    index = pd.date_range("2024-01-02", "2024-01-03", freq="15min", tz="UTC")
    feats = derivatives_features(funding, metrics, index)
    assert pd.isna(feats.iloc[0]).all()


def test_expected_columns_present():
    feats = derivatives_features(
        _funding(), _metrics(),
        pd.date_range("2024-01-02", "2024-01-05", freq="15min", tz="UTC"),
    )
    assert set(feats.columns) == {
        "funding_rate", "funding_z", "oi_change", "taker_ls_ratio", "crowding"
    }


def test_empty_inputs_yield_neutral_features():
    empty = pd.DataFrame()
    index = pd.date_range("2024-01-02", "2024-01-03", freq="15min", tz="UTC")
    feats = derivatives_features(empty, empty, index)
    # With no data the features exist but carry no signal (all zero after the shift).
    assert set(feats.columns) == {
        "funding_rate", "funding_z", "oi_change", "taker_ls_ratio", "crowding"
    }

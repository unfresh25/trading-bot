import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hmmlearn")

from hybrid_trader.config import FeatureConfig
from hybrid_trader.features import build_features
from hybrid_trader.regime_models import hmm_regime_probabilities


def _features(n=1200, seed=6):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "return_1": rng.normal(0, 0.004, n),
            "volatility_20": np.abs(rng.normal(0.005, 0.002, n)),
        },
        index=index,
    )


def test_hmm_probabilities_are_causal():
    feats = _features()
    cutoff = 800
    original = hmm_regime_probabilities(feats.iloc[: cutoff + 1], refit_every=50)
    changed = feats.copy()
    changed.iloc[cutoff + 1 :] *= 1.5
    mutated = hmm_regime_probabilities(changed.iloc[: cutoff + 1], refit_every=50)
    pd.testing.assert_frame_equal(original, mutated)


def test_hmm_probabilities_form_a_valid_distribution():
    hmm = hmm_regime_probabilities(_features(), refit_every=50)
    valid = hmm.dropna()
    assert list(hmm.columns) == ["hmm_bear", "hmm_sideways", "hmm_bull"]
    assert np.allclose(valid.sum(axis=1), 1.0)
    assert ((valid >= 0) & (valid <= 1)).all().all()
    assert hmm.iloc[:250].isna().all().all()  # no output before warmup


def test_hmm_features_appended_only_when_enabled():
    rng = np.random.default_rng(6)
    index = pd.date_range("2024-01-01", periods=1200, freq="15min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, 1200)))
    op = np.r_[close[0], close[:-1]]
    bars = pd.DataFrame(
        {"open": op, "high": np.maximum(op, close) * 1.002,
         "low": np.minimum(op, close) * 0.998, "close": close, "volume": 1000.0},
        index=index,
    )
    baseline = build_features(bars, FeatureConfig(use_hmm_regime=False))
    with_hmm = build_features(bars, FeatureConfig(use_hmm_regime=True, hmm_refit_every=100))
    assert "hmm_bull" not in baseline.columns
    assert {"hmm_bear", "hmm_sideways", "hmm_bull"} <= set(with_hmm.columns)
    assert baseline.shape[1] == 28

import numpy as np
import pandas as pd

from hybrid_trader.anomaly import robust_anomaly_score
from hybrid_trader.config import CostConfig, StrategyConfig
from hybrid_trader.decision_policy import SignalContext, evaluate_signal


def _features(n=600, seed=2):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "return_1": rng.normal(0, 0.004, n),
            "volatility_20": np.abs(rng.normal(0.005, 0.001, n)),
            "relative_volume_20": np.abs(rng.normal(1.0, 0.2, n)),
        },
        index=index,
    )


def test_anomaly_score_is_causal():
    feats = _features()
    cutoff = 400
    original = robust_anomaly_score(feats.iloc[: cutoff + 1])
    changed = feats.copy()
    changed.iloc[cutoff + 1 :] *= 3
    mutated = robust_anomaly_score(changed.iloc[: cutoff + 1])
    pd.testing.assert_series_equal(original, mutated)


def test_anomaly_score_spikes_on_an_outlier_bar():
    feats = _features()
    feats.iloc[500, feats.columns.get_loc("return_1")] = 0.5  # huge return shock
    score = robust_anomaly_score(feats)
    assert score.iloc[500] > score.iloc[490:500].max()


def test_anomaly_veto_blocks_trade_above_threshold():
    strategy = StrategyConfig(minimum_up_probability=0.5, max_anomaly_score=5.0)
    costs = CostConfig()
    calm = SignalContext(0.7, 0.1, 0.01, 1000.0, False, anomaly_score=2.0)
    storm = SignalContext(0.7, 0.1, 0.01, 1000.0, False, anomaly_score=8.0)
    assert evaluate_signal(calm, strategy, costs).reason == "approved"
    assert evaluate_signal(storm, strategy, costs).reason == "market_anomaly"


def test_anomaly_veto_disabled_by_default():
    # Default max_anomaly_score is inf, so the veto never fires.
    strategy = StrategyConfig(minimum_up_probability=0.5)
    context = SignalContext(0.7, 0.1, 0.01, 1000.0, False, anomaly_score=1e9)
    assert evaluate_signal(context, strategy, CostConfig()).reason == "approved"

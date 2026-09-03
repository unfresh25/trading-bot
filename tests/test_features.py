import pandas.testing as pdt

from hybrid_trader.features import build_features


def test_future_mutation_cannot_change_past_features(bars):
    cutoff = bars.index[300]
    original = build_features(bars).loc[:cutoff]
    changed = bars.copy()
    changed.loc[changed.index > cutoff, ["open", "high", "low", "close", "volume"]] *= 10
    mutated = build_features(changed).loc[:cutoff]
    pdt.assert_frame_equal(original, mutated)


def test_feature_count_is_deliberately_small(bars):
    assert 20 <= build_features(bars).shape[1] <= 30


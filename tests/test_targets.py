import pandas as pd
import pytest

from hybrid_trader.config import CostConfig, TargetConfig
from hybrid_trader.targets import build_targets


def test_target_uses_next_open_for_entry(bars):
    config = TargetConfig(horizon_bars=4)
    target = build_targets(bars, CostConfig(), config)
    location = 100
    expected = bars.open.iloc[location + 5] / bars.open.iloc[location + 1] - 1
    assert target.future_return.iloc[location] == pytest.approx(expected)


def test_last_horizon_plus_one_rows_have_no_label(bars):
    target = build_targets(bars, CostConfig(), TargetConfig(horizon_bars=4))
    assert target.label.tail(5).isna().all()


def test_threshold_covers_costs(bars):
    costs = CostConfig(commission_bps=1, spread_bps=4, slippage_bps=3)
    config = TargetConfig(cost_safety_multiple=2)
    target = build_targets(bars, costs, config)
    assert (target.threshold >= costs.round_trip_rate * 2).all()


def test_target_does_not_jump_over_a_missing_expected_bar(bars):
    signal_time = bars.index[100]
    expected_exit = signal_time + pd.Timedelta(minutes=75)
    bars_with_gap = bars.drop(expected_exit)

    target = build_targets(bars_with_gap, CostConfig(), TargetConfig(horizon_bars=4))

    assert pd.isna(target.loc[signal_time, "future_return"])
    assert pd.isna(target.loc[signal_time, "label"])

import pytest

from hybrid_trader.benchmarks import buy_and_hold, simple_strategy_benchmarks
from hybrid_trader.config import CostConfig


def test_buy_and_hold_enters_and_exits_with_costs(bars):
    result = buy_and_hold(bars, 500, CostConfig(spread_bps=4, slippage_bps=3))

    assert result.entry_time == bars.index[0]
    assert result.exit_time == bars.index[-1]
    assert result.cost > 0
    gross_return = bars.close.iloc[-1] / bars.open.iloc[0] - 1
    assert result.net_return < gross_return


def test_buy_and_hold_rejects_empty_dataset(bars):
    with pytest.raises(ValueError, match="empty"):
        buy_and_hold(bars.iloc[:0], 500, CostConfig())


def test_simple_strategies_report_costed_long_only_results(bars):
    results = simple_strategy_benchmarks(bars, 500, CostConfig())

    assert set(results) == {"momentum_sma_20", "mean_reversion_zscore_20"}
    assert all(result.trades >= 0 for result in results.values())
    assert all(result.cost >= 0 for result in results.values())

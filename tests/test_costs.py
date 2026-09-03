import pytest

from hybrid_trader.config import CostConfig
from hybrid_trader.costs import simulate_long_round_trip


def test_flat_market_loses_exactly_transaction_costs():
    config = CostConfig(commission_bps=1, spread_bps=4, slippage_bps=3)
    fill = simulate_long_round_trip(100, 100, 2, config)
    expected_commission = (fill.entry_fill + fill.exit_fill) * 2 / 10_000
    assert fill.commission == pytest.approx(expected_commission)
    assert fill.net_pnl < 0
    assert fill.net_pnl == pytest.approx((fill.exit_fill - fill.entry_fill) * 2 - expected_commission)


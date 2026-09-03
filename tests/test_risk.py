import pytest

from hybrid_trader.config import RiskConfig
from hybrid_trader.risk import size_long_position


def test_position_respects_risk_and_cash_caps():
    config = RiskConfig(initial_equity=500, risk_fraction=0.005, max_position_fraction=0.95)
    plan = size_long_position(500, 100, 1, config)
    assert plan.risk_amount <= 500 * 0.005
    assert plan.quantity * 100 <= 500 * 0.95
    assert plan.stop_price == pytest.approx(98)


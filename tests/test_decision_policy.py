from hybrid_trader.config import CostConfig, StrategyConfig
from hybrid_trader.contracts import Decision
from hybrid_trader.decision_policy import SignalContext, evaluate_signal


def test_zero_volume_is_a_hard_veto():
    result = evaluate_signal(
        SignalContext(0.8, 0.1, 0.01, 0.0, True), StrategyConfig(), CostConfig()
    )
    assert result.decision is Decision.NO_TRADE
    assert result.reason == "zero_volume"


def test_low_confidence_and_insufficient_edge_are_distinct_rejections():
    low_confidence = evaluate_signal(
        SignalContext(0.54, 0.2, 0.01, 10.0, False), StrategyConfig(), CostConfig()
    )
    low_edge = evaluate_signal(
        SignalContext(0.56, 0.44, 0.001, 10.0, False), StrategyConfig(), CostConfig()
    )
    assert low_confidence.reason == "low_model_confidence"
    assert low_edge.reason == "insufficient_edge_after_costs"


def test_positive_cost_adjusted_edge_approves_buy():
    result = evaluate_signal(
        SignalContext(0.75, 0.1, 0.01, 10.0, False), StrategyConfig(), CostConfig()
    )
    assert result.decision is Decision.BUY
    assert result.reason == "approved"
    assert result.expected_edge > 0

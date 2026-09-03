from __future__ import annotations

from dataclasses import dataclass

from .config import CostConfig, StrategyConfig
from .contracts import Decision


@dataclass(frozen=True)
class SignalContext:
    probability_up: float
    probability_down: float
    threshold: float
    volume: float
    reject_zero_volume: bool


@dataclass(frozen=True)
class DecisionEvaluation:
    decision: Decision
    reason: str
    expected_edge: float


def evaluate_signal(
    signal: SignalContext, strategy: StrategyConfig, costs: CostConfig
) -> DecisionEvaluation:
    edge = (signal.probability_up - signal.probability_down) * signal.threshold
    edge -= costs.round_trip_rate
    if signal.reject_zero_volume and signal.volume <= 0:
        return DecisionEvaluation(Decision.NO_TRADE, "zero_volume", edge)
    if signal.probability_up < strategy.minimum_up_probability:
        return DecisionEvaluation(Decision.NO_TRADE, "low_model_confidence", edge)
    if edge <= strategy.minimum_expected_edge:
        return DecisionEvaluation(Decision.NO_TRADE, "insufficient_edge_after_costs", edge)
    return DecisionEvaluation(Decision.BUY, "approved", edge)


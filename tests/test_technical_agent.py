import pandas as pd
import pytest
from pydantic import ValidationError

from hybrid_trader.agents.schemas import Direction, TechnicalSignal
from hybrid_trader.agents.technical_agent import technical_agent, technical_agent_scores
from hybrid_trader.config import CostConfig, StrategyConfig
from hybrid_trader.decision_policy import SignalContext, evaluate_signal


def _row(**overrides) -> pd.Series:
    base = {
        "distance_ema_20": 0.0,
        "trend_slope_20": 0.0,
        "macd_norm": 0.0,
        "rsi_14": 0.5,
        "atr_14_norm": 0.005,
    }
    base.update(overrides)
    return pd.Series(base)


def test_agent_abstains_when_a_required_indicator_is_missing():
    row = _row()
    del row["macd_norm"]
    assert technical_agent(row).abstain is True


def test_agent_abstains_on_balanced_evidence():
    # Flat indicators -> no net vote -> abstain, contributing nothing to the decision.
    signal = technical_agent(_row())
    assert signal.abstain is True
    assert signal.directional_score == 0.0


def test_agent_reads_a_clear_uptrend_as_bullish():
    signal = technical_agent(
        _row(distance_ema_20=0.01, trend_slope_20=0.001, macd_norm=0.003, rsi_14=0.35)
    )
    assert signal.direction is Direction.BULLISH
    assert signal.directional_score > 0
    assert signal.evidence  # traceable reasons attached


def test_agent_output_is_a_strict_validated_schema():
    signal = technical_agent(_row(distance_ema_20=0.02, trend_slope_20=0.002, macd_norm=0.004))
    # Round-trips through strict JSON and rejects out-of-range confidence.
    assert 0.0 <= signal.confidence <= 1.0
    with pytest.raises(ValidationError):
        TechnicalSignal(
            direction="bullish", confidence=1.5, trend_strength=0.5,
            risk="low", horizon="short_term",
        )


def test_agent_is_deterministic():
    row = _row(distance_ema_20=0.008, trend_slope_20=0.0005, macd_norm=0.002)
    a = technical_agent(row).model_dump()
    b = technical_agent(row).model_dump()
    assert a == b


def test_batch_scores_align_and_stay_in_range():
    features = pd.DataFrame([_row(distance_ema_20=x / 100) for x in range(-5, 6)])
    scores = technical_agent_scores(features)
    assert len(scores) == len(features)
    assert (scores.abs() <= 1.0).all()


def test_agent_disagreement_vetoes_a_long():
    strategy = StrategyConfig(minimum_up_probability=0.5)
    costs = CostConfig()
    agree = SignalContext(0.7, 0.1, 0.01, 1000.0, False, agent_score=0.5, agent_veto_below=-0.3)
    disagree = SignalContext(0.7, 0.1, 0.01, 1000.0, False, agent_score=-0.8, agent_veto_below=-0.3)
    assert evaluate_signal(agree, strategy, costs).reason == "approved"
    assert evaluate_signal(disagree, strategy, costs).reason == "agent_disagreement"

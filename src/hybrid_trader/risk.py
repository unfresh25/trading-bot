from __future__ import annotations

from dataclasses import dataclass

from .config import RiskConfig


@dataclass(frozen=True)
class PositionPlan:
    quantity: float
    stop_price: float
    risk_amount: float


def size_long_position(
    equity: float, entry_price: float, atr: float, config: RiskConfig
) -> PositionPlan:
    if min(equity, entry_price, atr) <= 0:
        raise ValueError("Equity, entry price and ATR must be positive")
    stop_distance = config.atr_stop_multiple * atr
    risk_amount = equity * config.risk_fraction
    risk_quantity = risk_amount / stop_distance
    cash_quantity = equity * config.max_position_fraction / entry_price
    quantity = max(0.0, min(risk_quantity, cash_quantity))
    return PositionPlan(quantity, entry_price - stop_distance, quantity * stop_distance)


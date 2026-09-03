from __future__ import annotations

from dataclasses import dataclass

from .config import CostConfig


@dataclass(frozen=True)
class FillPair:
    entry_fill: float
    exit_fill: float
    commission: float
    net_pnl: float


def simulate_long_round_trip(
    entry_mid: float, exit_mid: float, quantity: float, config: CostConfig
) -> FillPair:
    if min(entry_mid, exit_mid, quantity) < 0:
        raise ValueError("Prices and quantity cannot be negative")
    half_spread = config.spread_bps / 20_000
    slippage = config.slippage_bps / 10_000
    entry_fill = entry_mid * (1 + half_spread + slippage)
    exit_fill = exit_mid * (1 - half_spread - slippage)
    commission = (entry_fill + exit_fill) * quantity * config.commission_bps / 10_000
    net_pnl = (exit_fill - entry_fill) * quantity - commission
    return FillPair(entry_fill, exit_fill, commission, net_pnl)


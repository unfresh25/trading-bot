from __future__ import annotations

from dataclasses import dataclass

from .config import CostConfig
from .market_data import AssetClass


@dataclass(frozen=True)
class MarketProfile:
    symbol: str
    asset_class: AssetClass
    costs: CostConfig
    annualization_days: int
    reject_zero_volume: bool


def profile_for(symbol: str) -> MarketProfile:
    normalized = symbol.upper()
    if normalized == "BTC/USD":
        return MarketProfile(
            symbol=normalized,
            asset_class=AssetClass.CRYPTO,
            costs=CostConfig(commission_bps=0, spread_bps=8, slippage_bps=5),
            annualization_days=365,
            reject_zero_volume=True,
        )
    if normalized == "SPY":
        return MarketProfile(
            symbol=normalized,
            asset_class=AssetClass.STOCK,
            costs=CostConfig(commission_bps=0, spread_bps=1, slippage_bps=1),
            annualization_days=252,
            reject_zero_volume=False,
        )
    raise ValueError(f"No V0 market profile for {symbol!r}")


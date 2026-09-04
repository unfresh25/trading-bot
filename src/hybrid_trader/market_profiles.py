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
    atr_stop_multiple: float
    apply_intrabar_stop: bool


def profile_for(symbol: str) -> MarketProfile:
    normalized = symbol.upper()
    if normalized == "BTC/USD":
        # Stop ablation on locked BTC data: an executed intrabar stop destroys the
        # edge (dips trigger the stop before the price recovers). Size against a 2*ATR
        # stop but do not execute it; see the crypto-stop ablation experiment.
        return MarketProfile(
            symbol=normalized,
            asset_class=AssetClass.CRYPTO,
            costs=CostConfig(commission_bps=0, spread_bps=8, slippage_bps=5),
            annualization_days=365,
            reject_zero_volume=True,
            atr_stop_multiple=2.0,
            apply_intrabar_stop=False,
        )
    if normalized == "SPY":
        # For SPY the executed 2*ATR stop improves risk-adjusted return by bounding
        # tail losses, so it is kept on.
        return MarketProfile(
            symbol=normalized,
            asset_class=AssetClass.STOCK,
            costs=CostConfig(commission_bps=0, spread_bps=1, slippage_bps=1),
            annualization_days=252,
            reject_zero_volume=False,
            atr_stop_multiple=2.0,
            apply_intrabar_stop=True,
        )
    raise ValueError(f"No V0 market profile for {symbol!r}")


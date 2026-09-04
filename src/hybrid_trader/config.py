from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta


@dataclass(frozen=True)
class CostConfig:
    commission_bps: float = 0.0
    spread_bps: float = 2.0
    slippage_bps: float = 2.0

    @property
    def round_trip_rate(self) -> float:
        # Half spread is paid on each side, hence one full spread per round trip.
        return (2 * self.commission_bps + self.spread_bps + 2 * self.slippage_bps) / 10_000


@dataclass(frozen=True)
class TargetConfig:
    horizon_bars: int = 4
    bar_interval_minutes: int = 15
    volatility_window: int = 20
    volatility_multiple: float = 0.75
    cost_safety_multiple: float = 1.5
    minimum_threshold: float = 0.0005

    @property
    def bar_interval(self) -> timedelta:
        return timedelta(minutes=self.bar_interval_minutes)

    @property
    def holding_period(self) -> timedelta:
        return self.bar_interval * self.horizon_bars

    @property
    def label_span(self) -> timedelta:
        """Elapsed time from the signal close through the planned exit open."""
        return self.bar_interval + self.holding_period


@dataclass(frozen=True)
class RiskConfig:
    initial_equity: float = 500.0
    risk_fraction: float = 0.005
    max_position_fraction: float = 0.95
    max_drawdown: float = 0.10
    atr_stop_multiple: float = 2.0
    apply_intrabar_stop: bool = True
    """Whether the engine closes a trade intrabar when price pierces the ATR stop.

    ``atr_stop_multiple`` always drives position sizing. With this False the stop is
    used only for sizing and never executed (time-based exit), which lets ablation
    compare an executed stop against a sizing-only stop on the same footing.
    """


@dataclass(frozen=True)
class StrategyConfig:
    minimum_up_probability: float = 0.55
    minimum_expected_edge: float = 0.0
    cooldown_bars: int = 4
    long_only: bool = True


@dataclass(frozen=True)
class AppConfig:
    costs: CostConfig = field(default_factory=CostConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

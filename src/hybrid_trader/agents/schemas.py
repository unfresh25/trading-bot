from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Direction(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Horizon(StrEnum):
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


class Evidence(BaseModel):
    """A single traceable reason behind an agent's verdict."""

    model_config = ConfigDict(extra="forbid")

    indicator: str
    value: float
    interpretation: str


class TechnicalSignal(BaseModel):
    """Strict, auditable output of the technical agent (never an executable order).

    Mirrors the schema in the project plan. ``abstain`` is set when the evidence is
    insufficient, in which case ``direction`` is neutral and confidence is zero — the
    decision layer treats an abstaining agent as contributing nothing rather than noise.
    """

    model_config = ConfigDict(extra="forbid")

    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    trend_strength: float = Field(ge=0.0, le=1.0)
    risk: RiskLevel
    horizon: Horizon
    abstain: bool = False
    evidence: list[Evidence] = Field(default_factory=list)
    as_of: datetime | None = None

    @property
    def directional_score(self) -> float:
        """Signed conviction in [-1, 1]: positive bullish, negative bearish, 0 if abstaining."""
        if self.abstain or self.direction is Direction.NEUTRAL:
            return 0.0
        sign = 1.0 if self.direction is Direction.BULLISH else -1.0
        return sign * self.confidence

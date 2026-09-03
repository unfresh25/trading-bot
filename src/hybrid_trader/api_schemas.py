from __future__ import annotations

from datetime import datetime
from typing import Any, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class MarketBarRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_price_range(self) -> Self:
        if self.high < max(self.open, self.low, self.close):
            raise ValueError("high must be the greatest bar price")
        if self.low > min(self.open, self.high, self.close):
            raise ValueError("low must be the smallest bar price")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        return self


class CreateBacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1, max_length=20, pattern=r"^[A-Z0-9/.-]+$")
    bars: list[MarketBarRequest] = Field(min_length=1)

    def to_frame(self) -> pd.DataFrame:
        records = [bar.model_dump() for bar in self.bars]
        return pd.DataFrame.from_records(records).set_index("timestamp").sort_index()


class BacktestResource(BaseModel):
    id: str
    symbol: str
    metrics: dict[str, float]
    prediction_count: int
    trade_count: int


class HealthResource(BaseModel):
    status: str
    live_trading: bool


class ErrorResource(BaseModel):
    error: str
    message: str
    details: list[dict[str, Any]]
    timestamp: datetime
    path: str


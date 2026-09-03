from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import pandas as pd


class Decision(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NO_TRADE = "NO_TRADE"


class DataSource(Protocol):
    def bars(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...


class ProbabilisticModel(Protocol):
    @property
    def version(self) -> str: ...
    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None: ...
    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame: ...


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    cash: float
    drawdown: float


class Broker(ABC):
    """Execution boundary. V0 intentionally ships without a live implementation."""

    @abstractmethod
    def account(self) -> AccountSnapshot: ...

    @abstractmethod
    def positions(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def submit_order(self, request: dict[str, Any]) -> str: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> None: ...

    @abstractmethod
    def fills(self, order_id: str) -> list[dict[str, Any]]: ...


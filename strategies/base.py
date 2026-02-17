from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"


@dataclass
class Signal:
    timestamp: pd.Timestamp
    signal_type: SignalType
    symbol: str
    price: float
    quantity: int = 1
    stop_loss: float | None = None
    take_profit: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    All strategies implement this interface.
    Strategies are pure functions: they receive data and return signals.
    They do NOT manage positions or P&L.
    """

    name: str = "base"

    @abstractmethod
    def configure(self, params: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        ...

    def required_indicators(self) -> list[str]:
        return []

    def parameter_space(self) -> dict[str, list[Any]]:
        return {}

    def get_params(self) -> dict[str, Any]:
        return {}

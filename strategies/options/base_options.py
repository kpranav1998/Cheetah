from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from strategies.base import BaseStrategy, Signal


@dataclass
class OptionsLeg:
    option_type: str  # "CE" or "PE"
    strike: float
    expiry: date
    action: str  # "BUY" or "SELL"
    lots: int = 1


@dataclass
class OptionsSignal(Signal):
    legs: list[OptionsLeg] = field(default_factory=list)
    strategy_type: str = ""
    underlying_price: float = 0.0
    max_profit: float | None = None
    max_loss: float | None = None
    breakevens: list[float] = field(default_factory=list)


class BaseOptionsStrategy(BaseStrategy):
    """
    Options strategies differ from equity:
    - Emit OptionsSignal with multiple legs
    - Need options chain data
    - Have defined expiry
    - Non-linear P&L
    """

    @abstractmethod
    def select_strikes(
        self,
        underlying_price: float,
        options_chain: pd.DataFrame,
        params: dict[str, Any],
    ) -> list[OptionsLeg]:
        ...

    @abstractmethod
    def entry_condition(self, df: pd.DataFrame, chain: pd.DataFrame) -> bool:
        ...

    @abstractmethod
    def exit_condition(
        self,
        df: pd.DataFrame,
        chain: pd.DataFrame,
        entry_signal: OptionsSignal,
        current_pnl: float,
    ) -> bool:
        ...

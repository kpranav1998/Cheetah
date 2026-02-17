from __future__ import annotations

from typing import Any

import pandas as pd

from strategies.base import BaseStrategy, Signal, SignalType
from strategies.registry import register


@register
class RSIStrategy(BaseStrategy):
    """Buy when RSI crosses above oversold, sell when RSI crosses below overbought."""
    name = "rsi"

    def __init__(self):
        self.period = 14
        self.oversold = 30
        self.overbought = 70

    def configure(self, params: dict[str, Any]) -> None:
        self.period = params.get("period", self.period)
        self.oversold = params.get("oversold", self.oversold)
        self.overbought = params.get("overbought", self.overbought)

    def required_indicators(self) -> list[str]:
        return [f"rsi_{self.period}"]

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        rsi_col = df[f"rsi_{self.period}"]

        # Buy when RSI crosses above oversold level
        buy_signal = (rsi_col > self.oversold) & (rsi_col.shift(1) <= self.oversold)
        # Sell when RSI crosses below overbought level
        sell_signal = (rsi_col < self.overbought) & (rsi_col.shift(1) >= self.overbought)

        symbol = df.attrs.get("symbol", "")
        signals: list[Signal] = []

        for idx in df.index[buy_signal]:
            signals.append(Signal(timestamp=idx, signal_type=SignalType.BUY, symbol=symbol, price=df.loc[idx, "close"]))
        for idx in df.index[sell_signal]:
            signals.append(Signal(timestamp=idx, signal_type=SignalType.SELL, symbol=symbol, price=df.loc[idx, "close"]))

        return sorted(signals, key=lambda s: s.timestamp)

    def parameter_space(self) -> dict[str, list[Any]]:
        return {"period": [10, 14, 21], "oversold": [20, 30], "overbought": [70, 80]}

    def get_params(self) -> dict[str, Any]:
        return {"period": self.period, "oversold": self.oversold, "overbought": self.overbought}

from __future__ import annotations

from typing import Any

import pandas as pd

from strategies.base import BaseStrategy, Signal, SignalType
from strategies.registry import register


@register
class BollingerStrategy(BaseStrategy):
    """Mean-reversion: buy at lower band, sell at upper band."""
    name = "bollinger"

    def __init__(self):
        self.period = 20
        self.std_dev = 2.0

    def configure(self, params: dict[str, Any]) -> None:
        self.period = params.get("period", self.period)
        self.std_dev = params.get("std_dev", self.std_dev)

    def required_indicators(self) -> list[str]:
        return ["bollinger"]

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        close = df["close"]
        upper = df["bb_upper"]
        lower = df["bb_lower"]

        # Buy when price crosses below lower band then comes back above
        buy_signal = (close > lower) & (close.shift(1) <= lower.shift(1))
        # Sell when price crosses above upper band then comes back below
        sell_signal = (close < upper) & (close.shift(1) >= upper.shift(1))

        symbol = df.attrs.get("symbol", "")
        signals: list[Signal] = []

        for idx in df.index[buy_signal]:
            signals.append(Signal(
                timestamp=idx, signal_type=SignalType.BUY, symbol=symbol,
                price=df.loc[idx, "close"],
                take_profit=df.loc[idx, "bb_middle"],
            ))
        for idx in df.index[sell_signal]:
            signals.append(Signal(
                timestamp=idx, signal_type=SignalType.SELL, symbol=symbol,
                price=df.loc[idx, "close"],
            ))

        return sorted(signals, key=lambda s: s.timestamp)

    def parameter_space(self) -> dict[str, list[Any]]:
        return {"period": [15, 20, 25], "std_dev": [1.5, 2.0, 2.5]}

    def get_params(self) -> dict[str, Any]:
        return {"period": self.period, "std_dev": self.std_dev}

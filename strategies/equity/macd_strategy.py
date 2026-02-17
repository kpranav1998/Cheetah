from __future__ import annotations

from typing import Any

import pandas as pd

from strategies.base import BaseStrategy, Signal, SignalType
from strategies.registry import register


@register
class MACDStrategy(BaseStrategy):
    name = "macd"

    def __init__(self):
        self.fast = 12
        self.slow = 26
        self.signal_period = 9

    def configure(self, params: dict[str, Any]) -> None:
        self.fast = params.get("fast", self.fast)
        self.slow = params.get("slow", self.slow)
        self.signal_period = params.get("signal_period", self.signal_period)

    def required_indicators(self) -> list[str]:
        return ["macd"]

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        hist = df["macd_hist"]
        # Signal when histogram crosses zero
        cross_up = (hist > 0) & (hist.shift(1) <= 0)
        cross_down = (hist < 0) & (hist.shift(1) >= 0)

        symbol = df.attrs.get("symbol", "")
        signals: list[Signal] = []

        for idx in df.index[cross_up]:
            signals.append(Signal(timestamp=idx, signal_type=SignalType.BUY, symbol=symbol, price=df.loc[idx, "close"]))
        for idx in df.index[cross_down]:
            signals.append(Signal(timestamp=idx, signal_type=SignalType.SELL, symbol=symbol, price=df.loc[idx, "close"]))

        return sorted(signals, key=lambda s: s.timestamp)

    def parameter_space(self) -> dict[str, list[Any]]:
        return {"fast": [8, 12], "slow": [21, 26], "signal_period": [7, 9]}

    def get_params(self) -> dict[str, Any]:
        return {"fast": self.fast, "slow": self.slow, "signal_period": self.signal_period}

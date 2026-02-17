from __future__ import annotations

from typing import Any

import pandas as pd

from strategies.base import BaseStrategy, Signal, SignalType
from strategies.registry import register


@register
class EMACrossoverStrategy(BaseStrategy):
    name = "ema_crossover"

    def __init__(self):
        self.fast_period = 12
        self.slow_period = 26

    def configure(self, params: dict[str, Any]) -> None:
        self.fast_period = params.get("fast_period", self.fast_period)
        self.slow_period = params.get("slow_period", self.slow_period)

    def required_indicators(self) -> list[str]:
        return [f"ema_{self.fast_period}", f"ema_{self.slow_period}"]

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        fast = df[f"ema_{self.fast_period}"]
        slow = df[f"ema_{self.slow_period}"]
        cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))

        symbol = df.attrs.get("symbol", "")
        signals: list[Signal] = []

        for idx in df.index[cross_up]:
            signals.append(Signal(timestamp=idx, signal_type=SignalType.BUY, symbol=symbol, price=df.loc[idx, "close"]))
        for idx in df.index[cross_down]:
            signals.append(Signal(timestamp=idx, signal_type=SignalType.SELL, symbol=symbol, price=df.loc[idx, "close"]))

        return sorted(signals, key=lambda s: s.timestamp)

    def parameter_space(self) -> dict[str, list[Any]]:
        return {"fast_period": [8, 12, 21], "slow_period": [21, 26, 50]}

    def get_params(self) -> dict[str, Any]:
        return {"fast_period": self.fast_period, "slow_period": self.slow_period}

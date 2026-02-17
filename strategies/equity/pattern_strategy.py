from __future__ import annotations

from typing import Any

import pandas as pd

from patterns.scanner import scan_patterns
from strategies.base import BaseStrategy, Signal, SignalType
from strategies.registry import register


@register
class PatternStrategy(BaseStrategy):
    """Generates buy/sell signals from chart pattern detection."""
    name = "pattern"

    def __init__(self):
        self.pattern_names: list[str] | None = None  # None = all patterns
        self.min_confidence = 0.5

    def configure(self, params: dict[str, Any]) -> None:
        self.pattern_names = params.get("patterns", self.pattern_names)
        self.min_confidence = params.get("min_confidence", self.min_confidence)

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        matches = scan_patterns(df, self.pattern_names)
        symbol = df.attrs.get("symbol", "")
        signals: list[Signal] = []

        for match in matches:
            if match.confidence < self.min_confidence:
                continue

            if match.direction == "bullish":
                signals.append(Signal(
                    timestamp=match.confirmation_timestamp,
                    signal_type=SignalType.BUY,
                    symbol=symbol,
                    price=df.loc[match.confirmation_timestamp, "close"],
                    stop_loss=match.stop_loss,
                    take_profit=match.target_price,
                    metadata={
                        "pattern": match.pattern_type,
                        "confidence": match.confidence,
                        "target": match.target_price,
                    },
                ))
            else:  # bearish
                signals.append(Signal(
                    timestamp=match.confirmation_timestamp,
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    price=df.loc[match.confirmation_timestamp, "close"],
                    stop_loss=match.stop_loss,
                    take_profit=match.target_price,
                    metadata={
                        "pattern": match.pattern_type,
                        "confidence": match.confidence,
                        "target": match.target_price,
                    },
                ))

        return sorted(signals, key=lambda s: s.timestamp)

    def parameter_space(self) -> dict[str, list[Any]]:
        return {"min_confidence": [0.3, 0.5, 0.7]}

    def get_params(self) -> dict[str, Any]:
        return {"patterns": self.pattern_names, "min_confidence": self.min_confidence}

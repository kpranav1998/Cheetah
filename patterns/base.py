from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class PatternMatch:
    pattern_type: str
    start_idx: int
    end_idx: int
    confirmation_timestamp: pd.Timestamp
    direction: str  # "bullish" or "bearish"
    target_price: float
    stop_loss: float
    confidence: float  # 0.0 to 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BasePatternDetector(ABC):
    """Scans a full DataFrame and returns all detected pattern occurrences."""

    name: str = "base_pattern"

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> list[PatternMatch]:
        ...

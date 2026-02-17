from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Trade:
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    quantity: int
    entry_time: datetime
    exit_time: datetime
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.direction == "LONG":
            self.pnl = (self.exit_price - self.entry_price) * self.quantity - self.commission
            self.pnl_pct = ((self.exit_price / self.entry_price) - 1) * 100
        else:
            self.pnl = (self.entry_price - self.exit_price) * self.quantity - self.commission
            self.pnl_pct = ((self.entry_price / self.exit_price) - 1) * 100

    @property
    def holding_bars(self) -> int:
        return 0  # Filled by engine

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

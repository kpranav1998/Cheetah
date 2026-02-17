from __future__ import annotations

import pandas as pd

from execution.kite_broker import KiteBroker
from utils.logger import get_logger

logger = get_logger(__name__)


class PositionTracker:
    """Track real-time positions and MTM P&L."""

    def __init__(self, broker: KiteBroker | None = None):
        self.broker = broker or KiteBroker()

    def get_open_positions(self) -> pd.DataFrame:
        positions = self.broker.get_positions()
        if positions.empty:
            return positions
        return positions[positions["quantity"] != 0]

    def get_pnl_summary(self) -> dict:
        return self.broker.get_pnl()

    def get_position_details(self) -> list[dict]:
        """Get formatted position details."""
        positions = self.get_open_positions()
        if positions.empty:
            return []

        details = []
        for _, pos in positions.iterrows():
            details.append({
                "symbol": pos.get("tradingsymbol", ""),
                "exchange": pos.get("exchange", ""),
                "quantity": pos.get("quantity", 0),
                "avg_price": pos.get("average_price", 0),
                "ltp": pos.get("last_price", 0),
                "pnl": pos.get("pnl", 0),
                "product": pos.get("product", ""),
            })
        return details

    def has_position(self, symbol: str) -> bool:
        positions = self.get_open_positions()
        if positions.empty:
            return False
        return symbol in positions["tradingsymbol"].values

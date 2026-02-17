from __future__ import annotations

from execution.kite_broker import KiteBroker
from execution.position_tracker import PositionTracker
from strategies.base import Signal
from utils.logger import get_logger

logger = get_logger(__name__)


class RiskManager:
    """Validate signals against risk limits before execution."""

    def __init__(
        self,
        broker: KiteBroker | None = None,
        max_position_size_pct: float = 20.0,
        max_daily_loss: float = 50_000.0,
        max_open_positions: int = 5,
    ):
        self.broker = broker or KiteBroker()
        self.tracker = PositionTracker(self.broker)
        self.max_position_size_pct = max_position_size_pct
        self.max_daily_loss = max_daily_loss
        self.max_open_positions = max_open_positions

    def validate(self, signal: Signal) -> tuple[bool, str]:
        """Check if a signal passes risk checks. Returns (is_valid, reason)."""
        # Check daily loss limit
        pnl = self.tracker.get_pnl_summary()
        if pnl["total"] < -self.max_daily_loss:
            return False, f"Daily loss limit exceeded: {pnl['total']:.0f}"

        # Check max open positions
        open_positions = self.tracker.get_open_positions()
        if len(open_positions) >= self.max_open_positions:
            return False, f"Max open positions ({self.max_open_positions}) reached"

        # Check position size against available margin
        margins = self.broker.get_margins()
        equity_margin = margins.get("equity", {})
        available = equity_margin.get("available", {}).get("live_balance", 0)

        order_value = signal.price * signal.quantity
        if available > 0 and (order_value / available * 100) > self.max_position_size_pct:
            return False, f"Position size {order_value:.0f} exceeds {self.max_position_size_pct}% of margin"

        return True, "OK"

    def calculate_position_size(
        self,
        price: float,
        stop_loss: float | None = None,
        risk_per_trade_pct: float = 1.0,
    ) -> int:
        """Calculate optimal position size based on risk per trade."""
        margins = self.broker.get_margins()
        equity_margin = margins.get("equity", {})
        available = equity_margin.get("available", {}).get("live_balance", 0)

        if available <= 0:
            return 0

        risk_amount = available * (risk_per_trade_pct / 100)

        if stop_loss and stop_loss > 0:
            risk_per_share = abs(price - stop_loss)
            if risk_per_share > 0:
                return int(risk_amount / risk_per_share)

        # Default: use percentage of capital
        max_value = available * (self.max_position_size_pct / 100)
        return int(max_value / price)

from __future__ import annotations

from config.instruments import get_lot_size
from execution.kite_broker import KiteBroker
from strategies.base import Signal, SignalType
from strategies.options.base_options import OptionsSignal
from utils.logger import get_logger

logger = get_logger(__name__)


class OrderManager:
    """Converts strategy Signals into broker orders."""

    def __init__(self, broker: KiteBroker | None = None):
        self.broker = broker or KiteBroker()

    def execute_signal(self, signal: Signal) -> list[str]:
        """Convert a Signal into one or more broker orders. Returns order IDs."""
        if isinstance(signal, OptionsSignal):
            return self._execute_options_signal(signal)
        else:
            return self._execute_equity_signal(signal)

    def _execute_equity_signal(self, signal: Signal) -> list[str]:
        transaction_type = "BUY" if signal.signal_type in (SignalType.BUY, SignalType.COVER) else "SELL"

        order_id = self.broker.place_order(
            tradingsymbol=signal.symbol,
            exchange="NSE",
            transaction_type=transaction_type,
            order_type="MARKET",
            quantity=signal.quantity,
            product="MIS",
        )
        logger.info(f"Equity order: {transaction_type} {signal.quantity} {signal.symbol} -> {order_id}")
        return [order_id]

    def _execute_options_signal(self, signal: OptionsSignal) -> list[str]:
        order_ids = []
        lot_size = get_lot_size(signal.symbol)

        for leg in signal.legs:
            # Build NFO tradingsymbol (e.g., NIFTY2530622500CE)
            tradingsymbol = self._build_nfo_symbol(signal.symbol, leg)

            order_id = self.broker.place_order(
                tradingsymbol=tradingsymbol,
                exchange="NFO",
                transaction_type=leg.action,
                order_type="MARKET",
                quantity=leg.lots * lot_size,
                product="NRML",
            )
            order_ids.append(order_id)
            logger.info(f"Options order: {leg.action} {tradingsymbol} -> {order_id}")

        return order_ids

    def _build_nfo_symbol(self, underlying: str, leg) -> str:
        """Build Zerodha NFO tradingsymbol format."""
        expiry = leg.expiry
        month_map = {1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
                     7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"}

        year = str(expiry.year)[2:]  # Last 2 digits
        month = month_map[expiry.month]
        day = f"{expiry.day:02d}"

        strike = int(leg.strike) if leg.strike == int(leg.strike) else leg.strike
        return f"{underlying}{year}{month}{day}{strike}{leg.option_type}"

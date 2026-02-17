from __future__ import annotations

import pandas as pd
from kiteconnect import KiteConnect

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class KiteBroker:
    """Thin wrapper around KiteConnect for order placement and portfolio management."""

    def __init__(self):
        self.kite = KiteConnect(api_key=settings.kite_api_key)
        if settings.kite_access_token:
            self.kite.set_access_token(settings.kite_access_token)

    def place_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        order_type: str,
        quantity: int,
        product: str = "MIS",
        price: float | None = None,
        trigger_price: float | None = None,
        tag: str = "trading_agent",
    ) -> str:
        """Place an order and return order_id."""
        params = {
            "tradingsymbol": tradingsymbol,
            "exchange": exchange,
            "transaction_type": transaction_type,  # BUY or SELL
            "order_type": order_type,  # MARKET, LIMIT, SL, SL-M
            "quantity": quantity,
            "product": product,  # MIS, CNC, NRML
            "variety": "regular",
            "tag": tag,
        }

        if price is not None:
            params["price"] = price
        if trigger_price is not None:
            params["trigger_price"] = trigger_price

        order_id = self.kite.place_order(**params)
        logger.info(f"Order placed: {order_id} | {transaction_type} {quantity} {tradingsymbol}")
        return order_id

    def modify_order(
        self,
        order_id: str,
        quantity: int | None = None,
        price: float | None = None,
        trigger_price: float | None = None,
        order_type: str | None = None,
    ) -> str:
        params = {"order_id": order_id, "variety": "regular"}
        if quantity is not None:
            params["quantity"] = quantity
        if price is not None:
            params["price"] = price
        if trigger_price is not None:
            params["trigger_price"] = trigger_price
        if order_type is not None:
            params["order_type"] = order_type

        return self.kite.modify_order(**params)

    def cancel_order(self, order_id: str) -> str:
        return self.kite.cancel_order(variety="regular", order_id=order_id)

    def get_orders(self) -> pd.DataFrame:
        orders = self.kite.orders()
        return pd.DataFrame(orders) if orders else pd.DataFrame()

    def get_positions(self) -> pd.DataFrame:
        positions = self.kite.positions()
        day = pd.DataFrame(positions.get("day", []))
        net = pd.DataFrame(positions.get("net", []))
        return net if not net.empty else day

    def get_holdings(self) -> pd.DataFrame:
        holdings = self.kite.holdings()
        return pd.DataFrame(holdings) if holdings else pd.DataFrame()

    def get_margins(self) -> dict:
        return self.kite.margins()

    def get_pnl(self) -> dict:
        """Get today's realized and unrealized P&L from positions."""
        positions = self.kite.positions()
        net = positions.get("net", [])

        realized = sum(p.get("realised", 0) for p in net)
        unrealized = sum(p.get("unrealised", 0) for p in net)

        return {
            "realized": realized,
            "unrealized": unrealized,
            "total": realized + unrealized,
        }

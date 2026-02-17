from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from backtesting.trade import Trade
from config.instruments import CHARGES


@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    direction: str = "LONG"  # "LONG" or "SHORT"

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.quantity


class Portfolio:
    def __init__(
        self,
        initial_capital: float,
        commission_pct: float = 0.03,
        slippage_pct: float = 0.01,
        charge_type: str = "equity_intraday",
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.charge_type = charge_type
        self.positions: dict[str, Position] = {}
        self.equity_history: list[tuple[datetime, float]] = []

    def _calculate_commission(self, price: float, quantity: int) -> float:
        turnover = price * quantity
        charges = CHARGES.get(self.charge_type, CHARGES["equity_intraday"])
        brokerage = min(charges["brokerage_per_order"], turnover * 0.03 / 100)
        gst = brokerage * charges["gst_on_brokerage_pct"] / 100
        exchange = turnover * charges["exchange_txn_pct"] / 100
        sebi = turnover * charges["sebi_per_crore"] / 10_000_000
        return brokerage + gst + exchange + sebi

    def open_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        timestamp: datetime,
        direction: str = "LONG",
    ) -> None:
        slipped_price = price * (1 + self.slippage_pct / 100) if direction == "LONG" else price * (1 - self.slippage_pct / 100)
        cost = slipped_price * quantity
        commission = self._calculate_commission(slipped_price, quantity)

        if cost + commission > self.cash:
            return  # Insufficient funds

        self.cash -= cost + commission
        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=slipped_price,
            entry_time=timestamp,
            direction=direction,
        )

    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
    ) -> Trade | None:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None

        direction = pos.direction
        slipped_price = price * (1 - self.slippage_pct / 100) if direction == "LONG" else price * (1 + self.slippage_pct / 100)
        proceeds = slipped_price * pos.quantity
        commission = self._calculate_commission(slipped_price, pos.quantity) + self._calculate_commission(pos.entry_price, pos.quantity)

        self.cash += proceeds

        return Trade(
            symbol=symbol,
            direction=direction,
            entry_price=pos.entry_price,
            exit_price=slipped_price,
            quantity=pos.quantity,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            commission=commission,
        )

    def get_position(self, symbol: str) -> Position | None:
        return self.positions.get(symbol)

    def total_equity(self, current_prices: dict[str, float]) -> float:
        positions_value = sum(
            current_prices.get(s, p.entry_price) * p.quantity
            for s, p in self.positions.items()
        )
        return self.cash + positions_value

    def record_equity(self, timestamp: datetime, current_prices: dict[str, float]) -> None:
        self.equity_history.append((timestamp, self.total_equity(current_prices)))

    def get_equity_curve(self) -> pd.Series:
        if not self.equity_history:
            return pd.Series(dtype=float)
        times, values = zip(*self.equity_history)
        return pd.Series(values, index=pd.DatetimeIndex(times), name="equity")

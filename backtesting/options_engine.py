from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from backtesting.engine import BacktestConfig
from backtesting.metrics import compute_metrics
from backtesting.result import BacktestResult
from backtesting.trade import Trade
from config.instruments import get_lot_size
from options_pricing.black_scholes import (
    call_price, put_price, call_price_vec, put_price_vec, delta_call_vec,
)
from strategies.options.base_options import BaseOptionsStrategy, OptionsLeg, OptionsSignal
from utils.nse_calendar import get_weekly_expiry


@dataclass
class OptionsPosition:
    legs: list[OptionsLeg]
    entry_prices: list[float]  # Premium paid/received per leg
    entry_time: pd.Timestamp
    entry_underlying: float
    lot_size: int
    net_premium: float  # Positive = credit, negative = debit


class OptionsBacktestEngine:
    """
    Options backtesting with synthetic chain generation via Black-Scholes.
    Bar-by-bar because of time decay and IV changes.
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.risk_free_rate = 0.065  # ~RBI repo rate

    def run(
        self,
        strategy: BaseOptionsStrategy,
        underlying_df: pd.DataFrame,
        default_iv: float = 0.15,
        underlying_symbol: str = "NIFTY",
        entry_frequency_days: int = 7,  # Enter new position every N days
    ) -> BacktestResult:
        lot_size = get_lot_size(underlying_symbol)
        cash = self.config.capital
        positions: list[OptionsPosition] = []
        trades: list[Trade] = []
        equity_history: list[tuple[pd.Timestamp, float]] = []
        last_entry_idx = -entry_frequency_days

        for i, (timestamp, bar) in enumerate(underlying_df.iterrows()):
            underlying_price = bar["close"]
            current_date = timestamp.date() if hasattr(timestamp, "date") else timestamp

            # Get current expiry
            expiry = get_weekly_expiry(current_date)
            dte = (expiry - current_date).days
            T = max(dte / 365.0, 1 / 365.0)

            # Build synthetic chain
            chain = self._build_chain(underlying_price, T, default_iv)
            chain.attrs["expiry"] = expiry

            # Mark-to-market and check exits for open positions
            for pos in list(positions):
                pos_dte = (pos.legs[0].expiry - current_date).days
                pos_T = max(pos_dte / 365.0, 0)

                # Check expiry
                if pos_dte <= 0:
                    pnl = self._settle_at_expiry(pos, underlying_price, lot_size)
                    trades.append(self._make_trade(pos, pnl, timestamp, underlying_price))
                    positions.remove(pos)
                    cash += pnl
                    continue

                # Calculate current position value
                current_value = self._price_position(pos.legs, underlying_price, pos_T, default_iv, lot_size)
                current_pnl = current_value + pos.net_premium * lot_size  # net_premium is per lot

                # Check exit condition
                entry_signal = OptionsSignal(
                    timestamp=pos.entry_time,
                    signal_type=None,
                    symbol=underlying_symbol,
                    price=pos.entry_underlying,
                    max_profit=abs(pos.net_premium) * lot_size if pos.net_premium > 0 else None,
                    max_loss=-abs(pos.net_premium) * lot_size if pos.net_premium > 0 else None,
                )

                if strategy.exit_condition(underlying_df.iloc[:i + 1], chain, entry_signal, current_pnl):
                    trades.append(self._make_trade(pos, current_pnl, timestamp, underlying_price))
                    positions.remove(pos)
                    cash += current_pnl

            # Check entry condition
            if (i - last_entry_idx) >= entry_frequency_days and dte >= 3:
                if strategy.entry_condition(underlying_df.iloc[:i + 1], chain):
                    legs = strategy.select_strikes(underlying_price, chain, strategy.get_params())

                    # Price the legs
                    entry_prices = []
                    net_premium = 0.0
                    for leg in legs:
                        premium = self._price_leg(leg, underlying_price, T, default_iv)
                        entry_prices.append(premium)
                        multiplier = -1 if leg.action == "BUY" else 1
                        net_premium += multiplier * premium

                    # Check if we have enough capital
                    margin_required = abs(net_premium) * lot_size * 2  # Rough margin estimate
                    if margin_required < cash * 0.5:
                        positions.append(OptionsPosition(
                            legs=legs,
                            entry_prices=entry_prices,
                            entry_time=timestamp,
                            entry_underlying=underlying_price,
                            lot_size=lot_size,
                            net_premium=net_premium,
                        ))
                        last_entry_idx = i

            # Record equity
            open_pnl = 0.0
            for pos in positions:
                pos_dte = (pos.legs[0].expiry - current_date).days
                pos_T = max(pos_dte / 365.0, 0)
                current_value = self._price_position(pos.legs, underlying_price, pos_T, default_iv, lot_size)
                open_pnl += current_value + pos.net_premium * lot_size

            equity_history.append((timestamp, cash + open_pnl))

        # Close remaining positions at last bar
        if positions:
            last_ts = underlying_df.index[-1]
            last_price = underlying_df.iloc[-1]["close"]
            for pos in positions:
                pnl = self._settle_at_expiry(pos, last_price, lot_size)
                trades.append(self._make_trade(pos, pnl, last_ts, last_price))
                cash += pnl

        # Build equity curve
        times, values = zip(*equity_history) if equity_history else ([], [])
        equity_curve = pd.Series(values, index=pd.DatetimeIndex(times), name="equity")

        return BacktestResult(
            strategy_name=strategy.name,
            params=strategy.get_params(),
            symbol=underlying_df.attrs.get("symbol", ""),
            timeframe=underlying_df.attrs.get("timeframe", ""),
            start_date=underlying_df.index[0],
            end_date=underlying_df.index[-1],
            trades=trades,
            metrics=compute_metrics(trades, equity_curve, self.config.capital),
            equity_curve=equity_curve,
        )

    def _build_chain(self, spot: float, T: float, iv: float) -> pd.DataFrame:
        """Generate synthetic options chain using Black-Scholes."""
        # Strike interval: 50 for NIFTY-level, smaller for stocks
        interval = 50 if spot > 5000 else 10 if spot > 500 else 5
        min_strike = int(spot * 0.85 / interval) * interval
        max_strike = int(spot * 1.15 / interval) * interval + interval
        strikes = np.arange(min_strike, max_strike, interval, dtype=float)

        ce_prices = call_price_vec(spot, strikes, T, self.risk_free_rate, iv)
        pe_prices = put_price_vec(spot, strikes, T, self.risk_free_rate, iv)
        ce_deltas = delta_call_vec(spot, strikes, T, self.risk_free_rate, iv)

        return pd.DataFrame({
            "strike": strikes,
            "CE_price": ce_prices,
            "PE_price": pe_prices,
            "CE_delta": ce_deltas,
            "PE_delta": ce_deltas - 1,
        })

    def _price_leg(self, leg: OptionsLeg, spot: float, T: float, iv: float) -> float:
        if leg.option_type == "CE":
            return call_price(spot, leg.strike, T, self.risk_free_rate, iv)
        else:
            return put_price(spot, leg.strike, T, self.risk_free_rate, iv)

    def _price_position(
        self, legs: list[OptionsLeg], spot: float, T: float, iv: float, lot_size: int
    ) -> float:
        """Current MTM value of position (negative means position costs money to close)."""
        value = 0.0
        for leg in legs:
            premium = self._price_leg(leg, spot, T, iv)
            multiplier = 1 if leg.action == "BUY" else -1
            value += multiplier * premium * lot_size
        return value

    def _settle_at_expiry(self, pos: OptionsPosition, spot: float, lot_size: int) -> float:
        """Calculate P&L at expiry based on intrinsic value."""
        pnl = pos.net_premium * lot_size  # Start with premium received/paid

        for leg, entry_price in zip(pos.legs, pos.entry_prices):
            if leg.option_type == "CE":
                intrinsic = max(spot - leg.strike, 0)
            else:
                intrinsic = max(leg.strike - spot, 0)

            if leg.action == "BUY":
                pnl += (intrinsic - entry_price) * lot_size
            else:
                pnl += (entry_price - intrinsic) * lot_size

        # Subtract the double-counted net_premium
        pnl -= pos.net_premium * lot_size
        return pnl

    def _make_trade(
        self, pos: OptionsPosition, pnl: float, exit_time: pd.Timestamp, exit_price: float
    ) -> Trade:
        return Trade(
            symbol=f"OPTIONS_{pos.legs[0].strike}",
            direction="LONG" if pos.net_premium < 0 else "SHORT",
            entry_price=pos.entry_underlying,
            exit_price=exit_price,
            quantity=pos.lot_size,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            pnl=pnl,
            pnl_pct=(pnl / (abs(pos.net_premium) * pos.lot_size) * 100) if pos.net_premium != 0 else 0,
            metadata={
                "strategy_type": "options",
                "legs": [(l.option_type, l.strike, l.action) for l in pos.legs],
                "net_premium": pos.net_premium,
            },
        )

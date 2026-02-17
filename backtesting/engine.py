from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.metrics import compute_metrics
from backtesting.portfolio import Portfolio
from backtesting.result import BacktestResult
from backtesting.trade import Trade
from indicators import add_indicator
from strategies.base import BaseStrategy, SignalType


@dataclass
class BacktestConfig:
    capital: float = 1_000_000.0
    commission_pct: float = 0.03
    slippage_pct: float = 0.01
    position_size_pct: float = 95.0  # Use 95% of available capital per trade
    charge_type: str = "equity_intraday"


class BacktestEngine:
    """Core backtesting loop for equity and futures strategies."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(self, strategy: BaseStrategy, df: pd.DataFrame) -> BacktestResult:
        df = df.copy()

        # Phase 1: Pre-compute indicators (vectorized)
        for indicator_name in strategy.required_indicators():
            df = add_indicator(df, indicator_name)

        # Phase 2: Generate all signals (vectorized)
        signals = strategy.generate_signals(df)

        # Phase 3: Simulate execution
        portfolio = Portfolio(
            initial_capital=self.config.capital,
            commission_pct=self.config.commission_pct,
            slippage_pct=self.config.slippage_pct,
            charge_type=self.config.charge_type,
        )

        trades: list[Trade] = []
        signal_idx = 0
        sorted_signals = sorted(signals, key=lambda s: s.timestamp)

        for i, (timestamp, bar) in enumerate(df.iterrows()):
            current_price = bar["close"]
            symbol = df.attrs.get("symbol", "")

            # Check stop-loss / take-profit for open positions
            for sym, pos in list(portfolio.positions.items()):
                if pos.direction == "LONG":
                    # Check stop loss (using low of the bar)
                    if hasattr(bar, "low") and "low" in bar.index:
                        pass  # SL/TP checked below via signals

            # Process signals at this timestamp
            while signal_idx < len(sorted_signals) and sorted_signals[signal_idx].timestamp == timestamp:
                sig = sorted_signals[signal_idx]
                signal_idx += 1

                if sig.signal_type == SignalType.BUY:
                    if sig.symbol not in portfolio.positions:
                        # Position sizing: use configured % of available cash
                        available = portfolio.cash * (self.config.position_size_pct / 100)
                        qty = int(available / current_price)
                        if qty > 0:
                            portfolio.open_position(
                                sig.symbol or symbol, qty, current_price, timestamp
                            )

                elif sig.signal_type == SignalType.SELL:
                    target_sym = sig.symbol or symbol
                    trade = portfolio.close_position(target_sym, current_price, timestamp)
                    if trade:
                        trades.append(trade)

                elif sig.signal_type == SignalType.SHORT:
                    if sig.symbol not in portfolio.positions:
                        available = portfolio.cash * (self.config.position_size_pct / 100)
                        qty = int(available / current_price)
                        if qty > 0:
                            portfolio.open_position(
                                sig.symbol or symbol, qty, current_price, timestamp,
                                direction="SHORT",
                            )

                elif sig.signal_type == SignalType.COVER:
                    target_sym = sig.symbol or symbol
                    trade = portfolio.close_position(target_sym, current_price, timestamp)
                    if trade:
                        trades.append(trade)

            # Record equity
            prices = {s: bar["close"] for s in portfolio.positions}
            portfolio.record_equity(timestamp, prices)

        # Close any remaining positions at the last bar
        last_bar = df.iloc[-1]
        last_time = df.index[-1]
        for sym in list(portfolio.positions.keys()):
            trade = portfolio.close_position(sym, last_bar["close"], last_time)
            if trade:
                trades.append(trade)

        equity_curve = portfolio.get_equity_curve()

        return BacktestResult(
            strategy_name=strategy.name,
            params=strategy.get_params(),
            symbol=df.attrs.get("symbol", ""),
            timeframe=df.attrs.get("timeframe", ""),
            start_date=df.index[0],
            end_date=df.index[-1],
            trades=trades,
            metrics=compute_metrics(trades, equity_curve, self.config.capital),
            equity_curve=equity_curve,
        )

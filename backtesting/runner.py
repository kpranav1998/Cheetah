from __future__ import annotations

from datetime import datetime
from itertools import product
from typing import Any

import pandas as pd

from backtesting.engine import BacktestConfig, BacktestEngine
from backtesting.result import BacktestResult
from config.settings import settings
from config.timeframes import Timeframe
from data.data_manager import DataManager
from strategies.registry import get_strategy
from utils.logger import get_logger

logger = get_logger(__name__)


class BacktestRunner:
    """High-level orchestrator for backtesting."""

    def __init__(self, data_manager: DataManager | None = None, config: BacktestConfig | None = None):
        self.data_manager = data_manager or DataManager()
        self.config = config or BacktestConfig(
            capital=settings.default_capital,
            commission_pct=settings.default_commission_pct,
            slippage_pct=settings.default_slippage_pct,
        )

    def run_single(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        params: dict[str, Any] | None = None,
    ) -> BacktestResult:
        logger.info(f"Running {strategy_name} on {symbol} ({timeframe.value}) from {start} to {end}")

        strategy = get_strategy(strategy_name)
        if params:
            strategy.configure(params)

        df = self.data_manager.get_ohlcv(symbol, timeframe, start, end)
        if df.empty:
            raise ValueError(f"No data available for {symbol} from {start} to {end}")

        engine = BacktestEngine(self.config)
        result = engine.run(strategy, df)

        # Save result
        filename = f"{strategy_name}_{symbol}_{timeframe.value}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.json"
        result.save(settings.results_dir / filename)

        logger.info(
            f"Backtest complete: {result.metrics.total_trades} trades, "
            f"Return={result.metrics.total_return_pct:.2f}%, "
            f"Sharpe={result.metrics.sharpe_ratio:.2f}"
        )
        return result

    def run_parameter_sweep(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> list[BacktestResult]:
        strategy = get_strategy(strategy_name)
        param_space = strategy.parameter_space()

        if not param_space:
            return [self.run_single(strategy_name, symbol, timeframe, start, end)]

        keys = list(param_space.keys())
        values = list(param_space.values())
        results = []

        for combo in product(*values):
            params = dict(zip(keys, combo))
            # Skip invalid combos (e.g., fast >= slow for crossover)
            if "fast_period" in params and "slow_period" in params:
                if params["fast_period"] >= params["slow_period"]:
                    continue

            try:
                result = self.run_single(strategy_name, symbol, timeframe, start, end, params)
                results.append(result)
            except Exception as e:
                logger.warning(f"Sweep failed for {params}: {e}")

        results.sort(key=lambda r: r.metrics.sharpe_ratio, reverse=True)
        return results

    def run_multi_symbol(
        self,
        strategy_name: str,
        symbols: list[str],
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        params: dict[str, Any] | None = None,
    ) -> dict[str, BacktestResult]:
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.run_single(
                    strategy_name, symbol, timeframe, start, end, params
                )
            except Exception as e:
                logger.warning(f"Backtest failed for {symbol}: {e}")
        return results

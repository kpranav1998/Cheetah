from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backtesting.metrics import BacktestMetrics
from backtesting.trade import Trade


@dataclass
class BacktestResult:
    strategy_name: str
    params: dict[str, Any]
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    trades: list[Trade]
    metrics: BacktestMetrics
    equity_curve: pd.Series

    def top_winners(self, n: int = 5) -> list[Trade]:
        return sorted(self.trades, key=lambda t: t.pnl, reverse=True)[:n]

    def top_losers(self, n: int = 5) -> list[Trade]:
        return sorted(self.trades, key=lambda t: t.pnl)[:n]

    def monthly_returns(self) -> pd.Series:
        if self.equity_curve.empty:
            return pd.Series(dtype=float)
        monthly = self.equity_curve.resample("ME").last()
        return monthly.pct_change().dropna() * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "params": self.params,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "total_trades": len(self.trades),
            "metrics": {
                "total_return_pct": self.metrics.total_return_pct,
                "cagr_pct": self.metrics.cagr_pct,
                "sharpe_ratio": self.metrics.sharpe_ratio,
                "sortino_ratio": self.metrics.sortino_ratio,
                "max_drawdown_pct": self.metrics.max_drawdown_pct,
                "win_rate_pct": self.metrics.win_rate_pct,
                "profit_factor": self.metrics.profit_factor,
                "total_trades": self.metrics.total_trades,
                "expectancy": self.metrics.expectancy,
            },
            "trades": [
                {
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "entry_time": str(t.entry_time),
                    "exit_time": str(t.exit_time),
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                }
                for t in self.trades
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def summary_for_llm(self) -> str:
        winners = self.top_winners(5)
        losers = self.top_losers(5)
        monthly = self.monthly_returns()

        lines = [
            f"Strategy: {self.strategy_name}",
            f"Symbol: {self.symbol} | Timeframe: {self.timeframe}",
            f"Period: {self.start_date} to {self.end_date}",
            f"Parameters: {json.dumps(self.params)}",
            "",
            "=== Performance Metrics ===",
            self.metrics.to_table_string(),
            "",
            "=== Top 5 Winners ===",
        ]
        for t in winners:
            lines.append(f"  {t.entry_time} -> {t.exit_time}: PnL={t.pnl:.2f} ({t.pnl_pct:.2f}%)")
        lines.append("")
        lines.append("=== Top 5 Losers ===")
        for t in losers:
            lines.append(f"  {t.entry_time} -> {t.exit_time}: PnL={t.pnl:.2f} ({t.pnl_pct:.2f}%)")

        if not monthly.empty:
            lines.append("")
            lines.append("=== Monthly Returns ===")
            for dt, ret in monthly.items():
                lines.append(f"  {dt.strftime('%Y-%m')}: {ret:.2f}%")

        return "\n".join(lines)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtesting.trade import Trade


@dataclass
class BacktestMetrics:
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_return_pct: float = 0.0
    avg_winner_pct: float = 0.0
    avg_loser_pct: float = 0.0
    avg_holding_period_bars: int = 0
    expectancy: float = 0.0

    def to_table_string(self) -> str:
        lines = [
            f"Total Return:       {self.total_return_pct:>10.2f}%",
            f"CAGR:               {self.cagr_pct:>10.2f}%",
            f"Sharpe Ratio:       {self.sharpe_ratio:>10.2f}",
            f"Sortino Ratio:      {self.sortino_ratio:>10.2f}",
            f"Max Drawdown:       {self.max_drawdown_pct:>10.2f}%",
            f"Max DD Duration:    {self.max_drawdown_duration_days:>10d} days",
            f"Win Rate:           {self.win_rate_pct:>10.2f}%",
            f"Profit Factor:      {self.profit_factor:>10.2f}",
            f"Total Trades:       {self.total_trades:>10d}",
            f"Avg Trade Return:   {self.avg_trade_return_pct:>10.2f}%",
            f"Avg Winner:         {self.avg_winner_pct:>10.2f}%",
            f"Avg Loser:          {self.avg_loser_pct:>10.2f}%",
            f"Expectancy:         {self.expectancy:>10.2f}",
        ]
        return "\n".join(lines)


def compute_metrics(
    trades: list[Trade],
    equity_curve: pd.Series,
    initial_capital: float,
) -> BacktestMetrics:
    m = BacktestMetrics()
    m.total_trades = len(trades)

    if not trades:
        return m

    # Returns
    final_equity = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
    m.total_return_pct = ((final_equity / initial_capital) - 1) * 100

    # CAGR
    if len(equity_curve) > 1:
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if days > 0:
            m.cagr_pct = ((final_equity / initial_capital) ** (365.0 / days) - 1) * 100

    # Drawdown
    if len(equity_curve) > 0:
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        m.max_drawdown_pct = abs(drawdown.min()) * 100

        # Drawdown duration
        in_dd = drawdown < 0
        if in_dd.any():
            dd_groups = (~in_dd).cumsum()
            dd_durations = in_dd.groupby(dd_groups).sum()
            m.max_drawdown_duration_days = int(dd_durations.max())

    # Sharpe & Sortino (daily returns, annualized)
    if len(equity_curve) > 1:
        daily_returns = equity_curve.pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            m.sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            downside = daily_returns[daily_returns < 0]
            if len(downside) > 0 and downside.std() > 0:
                m.sortino_ratio = (daily_returns.mean() / downside.std()) * np.sqrt(252)

    # Trade statistics
    pnls = [t.pnl for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]
    winners = [t for t in trades if t.is_winner]
    losers = [t for t in trades if not t.is_winner]

    m.win_rate_pct = (len(winners) / len(trades)) * 100 if trades else 0

    gross_profit = sum(t.pnl for t in winners)
    gross_loss = abs(sum(t.pnl for t in losers))
    m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    m.avg_trade_return_pct = np.mean(pnl_pcts) if pnl_pcts else 0
    m.avg_winner_pct = np.mean([t.pnl_pct for t in winners]) if winners else 0
    m.avg_loser_pct = np.mean([t.pnl_pct for t in losers]) if losers else 0

    # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    win_rate = len(winners) / len(trades) if trades else 0
    loss_rate = 1 - win_rate
    avg_win = np.mean([t.pnl for t in winners]) if winners else 0
    avg_loss = abs(np.mean([t.pnl for t in losers])) if losers else 0
    m.expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

    return m

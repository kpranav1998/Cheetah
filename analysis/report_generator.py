from __future__ import annotations

from pathlib import Path
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtesting.result import BacktestResult
from config.settings import settings


class ReportGenerator:
    """Generate HTML backtest reports with charts."""

    def generate(
        self,
        result: BacktestResult,
        llm_analysis: str = "",
        output_path: Path | None = None,
    ) -> Path:
        if output_path is None:
            filename = f"{result.strategy_name}_{result.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            output_path = settings.reports_dir / filename

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build charts
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Equity Curve", "Drawdown", "Trade P&L Distribution"),
            row_heights=[0.4, 0.3, 0.3],
            vertical_spacing=0.08,
        )

        # Equity curve
        if not result.equity_curve.empty:
            fig.add_trace(
                go.Scatter(
                    x=result.equity_curve.index,
                    y=result.equity_curve.values,
                    name="Equity",
                    line=dict(color="blue"),
                ),
                row=1, col=1,
            )

            # Drawdown
            peak = result.equity_curve.cummax()
            drawdown = (result.equity_curve - peak) / peak * 100
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    fill="tozeroy",
                    name="Drawdown %",
                    line=dict(color="red"),
                ),
                row=2, col=1,
            )

        # Trade P&L histogram
        if result.trades:
            pnls = [t.pnl for t in result.trades]
            colors = ["green" if p > 0 else "red" for p in pnls]
            fig.add_trace(
                go.Bar(
                    x=list(range(len(pnls))),
                    y=pnls,
                    marker_color=colors,
                    name="Trade P&L",
                ),
                row=3, col=1,
            )

        fig.update_layout(
            title=f"Backtest Report: {result.strategy_name} on {result.symbol}",
            height=900,
            showlegend=False,
        )

        chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

        # Build full HTML report
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report: {result.strategy_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a1a1a; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #333; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f8f8; font-weight: 600; }}
        .positive {{ color: #4CAF50; }}
        .negative {{ color: #f44336; }}
        .analysis {{ background: #f9f9f9; padding: 20px; border-radius: 6px; margin: 20px 0; white-space: pre-wrap; line-height: 1.6; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Backtest Report: {result.strategy_name} on {result.symbol}</h1>
    <p><strong>Period:</strong> {result.start_date} to {result.end_date} | <strong>Timeframe:</strong> {result.timeframe}</p>
    <p><strong>Parameters:</strong> {result.params}</p>

    <h2>Performance Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Return</td><td class="{'positive' if result.metrics.total_return_pct >= 0 else 'negative'}">{result.metrics.total_return_pct:.2f}%</td></tr>
        <tr><td>CAGR</td><td>{result.metrics.cagr_pct:.2f}%</td></tr>
        <tr><td>Sharpe Ratio</td><td>{result.metrics.sharpe_ratio:.2f}</td></tr>
        <tr><td>Sortino Ratio</td><td>{result.metrics.sortino_ratio:.2f}</td></tr>
        <tr><td>Max Drawdown</td><td class="negative">{result.metrics.max_drawdown_pct:.2f}%</td></tr>
        <tr><td>Win Rate</td><td>{result.metrics.win_rate_pct:.2f}%</td></tr>
        <tr><td>Profit Factor</td><td>{result.metrics.profit_factor:.2f}</td></tr>
        <tr><td>Total Trades</td><td>{result.metrics.total_trades}</td></tr>
        <tr><td>Expectancy</td><td>{result.metrics.expectancy:.2f}</td></tr>
    </table>

    <h2>Charts</h2>
    {chart_html}

    {"<h2>LLM Analysis</h2><div class='analysis'>" + llm_analysis + "</div>" if llm_analysis else ""}
</div>
</body>
</html>"""

        with open(output_path, "w") as f:
            f.write(html)

        return output_path

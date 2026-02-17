from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

# Import all strategies to trigger registration
import strategies.equity.sma_crossover  # noqa: F401
import strategies.equity.ema_crossover  # noqa: F401
import strategies.equity.macd_strategy  # noqa: F401
import strategies.equity.bollinger_strategy  # noqa: F401
import strategies.equity.rsi_strategy  # noqa: F401
import strategies.equity.pattern_strategy  # noqa: F401
import strategies.options.iron_condor  # noqa: F401
import strategies.options.bull_call_spread  # noqa: F401
import strategies.options.bear_put_spread  # noqa: F401
import strategies.options.straddle  # noqa: F401
import strategies.options.strangle  # noqa: F401
import strategies.options.covered_call  # noqa: F401
import strategies.options.protective_put  # noqa: F401

from backtesting.engine import BacktestConfig
from backtesting.options_engine import OptionsBacktestEngine
from backtesting.runner import BacktestRunner
from config.timeframes import Timeframe
from data.data_manager import DataManager
from strategies.options.base_options import BaseOptionsStrategy
from strategies.registry import get_strategy, list_strategies

app = typer.Typer(help="Trading Agent - Backtest & Trade Indian Markets")
console = Console()


def _display_result(result, strategy_name, symbol):
    table = Table(title=f"Backtest Results: {strategy_name} on {symbol}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Return", f"{result.metrics.total_return_pct:.2f}%")
    table.add_row("CAGR", f"{result.metrics.cagr_pct:.2f}%")
    table.add_row("Sharpe Ratio", f"{result.metrics.sharpe_ratio:.2f}")
    table.add_row("Sortino Ratio", f"{result.metrics.sortino_ratio:.2f}")
    table.add_row("Max Drawdown", f"{result.metrics.max_drawdown_pct:.2f}%")
    table.add_row("Win Rate", f"{result.metrics.win_rate_pct:.2f}%")
    table.add_row("Profit Factor", f"{result.metrics.profit_factor:.2f}")
    table.add_row("Total Trades", str(result.metrics.total_trades))
    table.add_row("Expectancy", f"{result.metrics.expectancy:.2f}")
    console.print(table)


@app.command()
def backtest(
    strategy: str = typer.Option(..., help="Strategy name (e.g., sma_crossover)"),
    symbol: str = typer.Option(..., help="Symbol (e.g., RELIANCE.NS, ^NSEI)"),
    timeframe: str = typer.Option("1d", help="Timeframe: 1m, 5m, 15m, 1h, 1d, 1wk"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    fast_period: int = typer.Option(None, help="Fast MA period (for crossover strategies)"),
    slow_period: int = typer.Option(None, help="Slow MA period (for crossover strategies)"),
):
    """Run a backtest for a given strategy and symbol."""
    tf = Timeframe(timeframe)
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    params = {}
    if fast_period is not None:
        params["fast_period"] = fast_period
    if slow_period is not None:
        params["slow_period"] = slow_period

    # Check if it's an options strategy
    strat = get_strategy(strategy)
    if isinstance(strat, BaseOptionsStrategy):
        if params:
            strat.configure(params)
        dm = DataManager()
        df = dm.get_ohlcv(symbol, tf, start_dt, end_dt)
        engine = OptionsBacktestEngine(BacktestConfig())
        result = engine.run(strat, df, underlying_symbol=symbol.replace(".NS", "").replace("^", ""))
    else:
        runner = BacktestRunner()
        result = runner.run_single(strategy, symbol, timeframe=tf, start=start_dt, end=end_dt, params=params or None)

    _display_result(result, strategy, symbol)


@app.command()
def sweep(
    strategy: str = typer.Option(..., help="Strategy name"),
    symbol: str = typer.Option(..., help="Symbol"),
    timeframe: str = typer.Option("1d", help="Timeframe"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
):
    """Run parameter sweep for a strategy."""
    tf = Timeframe(timeframe)
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    runner = BacktestRunner()
    results = runner.run_parameter_sweep(strategy, symbol, tf, start_dt, end_dt)

    table = Table(title=f"Parameter Sweep: {strategy} on {symbol} (Top 10)")
    table.add_column("Params", style="cyan")
    table.add_column("Return %", style="green")
    table.add_column("Sharpe", style="yellow")
    table.add_column("Win Rate %", style="blue")
    table.add_column("Trades", style="white")

    for r in results[:10]:
        table.add_row(
            str(r.params),
            f"{r.metrics.total_return_pct:.2f}",
            f"{r.metrics.sharpe_ratio:.2f}",
            f"{r.metrics.win_rate_pct:.2f}",
            str(r.metrics.total_trades),
        )
    console.print(table)


@app.command()
def analyze(
    strategy: str = typer.Option(..., help="Strategy name"),
    symbol: str = typer.Option(..., help="Symbol"),
    timeframe: str = typer.Option("1d", help="Timeframe"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    report: bool = typer.Option(False, help="Generate HTML report"),
):
    """Run backtest + LLM analysis."""
    tf = Timeframe(timeframe)
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    runner = BacktestRunner()
    result = runner.run_single(strategy, symbol, tf, start_dt, end_dt)
    _display_result(result, strategy, symbol)

    console.print("\n[yellow]Running LLM analysis...[/yellow]")
    from analysis.result_analyzer import BacktestResultAnalyzer
    analyzer = BacktestResultAnalyzer()
    analysis = analyzer.analyze(result)
    console.print(f"\n[bold]LLM Analysis:[/bold]\n{analysis}")

    if report:
        from analysis.report_generator import ReportGenerator
        gen = ReportGenerator()
        path = gen.generate(result, llm_analysis=analysis)
        console.print(f"\n[green]Report saved to: {path}[/green]")


@app.command(name="list")
def list_cmd():
    """List available strategies."""
    console.print("[bold]Available Strategies:[/bold]")
    for name in list_strategies():
        console.print(f"  - {name}")


@app.command()
def positions():
    """Show current live positions from Zerodha."""
    from execution.position_tracker import PositionTracker
    tracker = PositionTracker()
    details = tracker.get_position_details()

    if not details:
        console.print("No open positions.")
        return

    table = Table(title="Open Positions")
    table.add_column("Symbol")
    table.add_column("Qty")
    table.add_column("Avg Price")
    table.add_column("LTP")
    table.add_column("P&L")

    for d in details:
        pnl_style = "green" if d["pnl"] >= 0 else "red"
        table.add_row(
            d["symbol"], str(d["quantity"]),
            f"{d['avg_price']:.2f}", f"{d['ltp']:.2f}",
            f"[{pnl_style}]{d['pnl']:.2f}[/{pnl_style}]",
        )
    console.print(table)


@app.command()
def scan(
    csv_file: str = typer.Argument(..., help="Path to CSV file with OHLCV data"),
    pattern: str = typer.Option(None, help="Specific pattern to scan (e.g., flag_pole, double_bottom). Default: all"),
):
    """Scan a CSV file for chart patterns (flag pole, double bottom, head & shoulders, etc.)."""
    from patterns.scanner import scan_patterns, DETECTOR_MAP

    path = Path(csv_file)
    if not path.exists():
        console.print(f"[red]File not found: {csv_file}[/red]")
        raise typer.Exit(1)

    # Load CSV
    df = pd.read_csv(path)

    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Try to find and set datetime index
    date_col = None
    for candidate in ["date", "datetime", "timestamp", "time"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    else:
        # If first column looks like dates, use it
        try:
            df.index = pd.to_datetime(df.iloc[:, 0])
            df = df.iloc[:, 1:]
        except Exception:
            console.print("[yellow]Warning: Could not detect date column. Using row index.[/yellow]")
            df.index = pd.RangeIndex(len(df))

    df = df.sort_index()

    # Validate required columns
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        console.print(f"[red]CSV missing required columns: {missing}[/red]")
        console.print(f"[dim]Found columns: {list(df.columns)}[/dim]")
        raise typer.Exit(1)

    console.print(f"[bold]Loaded {len(df)} rows from {path.name}[/bold]")
    console.print(f"[dim]Date range: {df.index[0]} to {df.index[-1]}[/dim]")
    console.print()

    # Run pattern detection
    pattern_names = [pattern] if pattern else None
    if pattern and pattern not in DETECTOR_MAP:
        console.print(f"[red]Unknown pattern: {pattern}[/red]")
        console.print(f"Available patterns: {', '.join(DETECTOR_MAP.keys())}")
        raise typer.Exit(1)

    console.print("[yellow]Scanning for patterns...[/yellow]")
    matches = scan_patterns(df, pattern_names)

    if not matches:
        console.print("[dim]No patterns detected.[/dim]")
        return

    console.print(f"\n[bold green]Found {len(matches)} pattern(s)![/bold green]\n")

    table = Table(title="Detected Patterns")
    table.add_column("#", style="dim")
    table.add_column("Pattern", style="cyan bold")
    table.add_column("Direction", style="white")
    table.add_column("Confirmed At", style="white")
    table.add_column("Target", style="green")
    table.add_column("Stop Loss", style="red")
    table.add_column("Confidence", style="yellow")

    for i, m in enumerate(matches, 1):
        dir_style = "green" if m.direction == "bullish" else "red"
        table.add_row(
            str(i),
            m.pattern_type.replace("_", " ").title(),
            f"[{dir_style}]{m.direction.upper()}[/{dir_style}]",
            str(m.confirmation_timestamp)[:19],
            f"{m.target_price:.2f}",
            f"{m.stop_loss:.2f}",
            f"{m.confidence:.0%}",
        )

    console.print(table)

    # Print detailed info for each match
    console.print("\n[bold]Details:[/bold]")
    for i, m in enumerate(matches, 1):
        dir_icon = "^" if m.direction == "bullish" else "v"
        console.print(
            f"\n  [{i}] {m.pattern_type.replace('_', ' ').title()} ({dir_icon} {m.direction})"
        )
        console.print(f"      Formed between bars {m.start_idx} - {m.end_idx}")
        console.print(f"      Confirmation: {m.confirmation_timestamp}")
        confirm_price = df.loc[m.confirmation_timestamp, "close"] if m.confirmation_timestamp in df.index else "N/A"
        console.print(f"      Entry price: {confirm_price}")
        console.print(f"      Target: {m.target_price:.2f} | Stop Loss: {m.stop_loss:.2f}")
        if m.metadata:
            for k, v in m.metadata.items():
                console.print(f"      {k}: {v}")


if __name__ == "__main__":
    app()

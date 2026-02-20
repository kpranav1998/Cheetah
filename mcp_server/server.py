"""MCP Server exposing trading agent tools via stdio transport."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import all strategies to trigger @register
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

from backtesting.runner import BacktestRunner
from config.timeframes import Timeframe
from data.data_manager import DataManager
from indicators import add_indicator
from patterns.scanner import scan_patterns, DETECTOR_MAP
from patterns.support_resistance import find_support_levels, find_resistance_levels
from strategies.registry import list_strategies
from utils.logger import get_logger, log_tool_call

logger = get_logger("mcp_server")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

mcp = FastMCP("TradingAgent")

_data_manager = DataManager()
_df_cache: dict[str, pd.DataFrame] = {}


def _cache_key_symbol(symbol: str, timeframe: str, start_date: str, end_date: str) -> str:
    return f"{symbol}:{timeframe}:{start_date}:{end_date}"


def _cache_key_csv(file_path: str) -> str:
    return f"csv:{file_path}"


def _load_df(
    symbol: str | None = None,
    timeframe: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    csv_path: str | None = None,
) -> pd.DataFrame:
    """Load a DataFrame from CSV or by fetching OHLCV data. Uses internal cache."""
    if csv_path:
        key = _cache_key_csv(csv_path)
        if key in _df_cache:
            return _df_cache[key]
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        for candidate in ("date", "datetime", "timestamp", "time"):
            if candidate in df.columns:
                df[candidate] = pd.to_datetime(df[candidate])
                df = df.set_index(candidate)
                break
        df = df.sort_index()
        _df_cache[key] = df
        return df

    if not symbol or not start_date or not end_date:
        raise ValueError("Must provide either csv_path or (symbol, start_date, end_date)")

    key = _cache_key_symbol(symbol, timeframe, start_date, end_date)
    if key in _df_cache:
        return _df_cache[key]

    tf = Timeframe(timeframe)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    df = _data_manager.get_ohlcv(symbol, tf, start_dt, end_dt)
    _df_cache[key] = df
    return df


def _pattern_match_to_dict(m) -> dict[str, Any]:
    return {
        "pattern_type": m.pattern_type,
        "start_idx": m.start_idx,
        "end_idx": m.end_idx,
        "confirmation_timestamp": str(m.confirmation_timestamp),
        "direction": m.direction,
        "target_price": round(m.target_price, 2),
        "stop_loss": round(m.stop_loss, 2),
        "confidence": round(m.confidence, 3),
        "metadata": {k: _serialize(v) for k, v in m.metadata.items()},
    }


def _serialize(v: Any) -> Any:
    if isinstance(v, (pd.Timestamp, datetime)):
        return str(v)
    if isinstance(v, float):
        return round(v, 4)
    return v


# ===================================================================
# DATA TOOLS
# ===================================================================

@mcp.tool()
@log_tool_call
async def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    start_date: str = "2023-01-01",
    end_date: str = "2025-01-01",
) -> str:
    """Fetch OHLCV (Open/High/Low/Close/Volume) price data for a symbol.

    Args:
        symbol: Ticker symbol (e.g. RELIANCE.NS, TCS.NS, ^NSEI for Nifty 50)
        timeframe: Candle timeframe — 1m, 5m, 15m, 30m, 1h, 1d, 1wk
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        JSON with row count, date range, basic stats, and last 5 rows.
    """
    df = _load_df(symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date)
    if df.empty:
        return json.dumps({"error": f"No data found for {symbol}"})

    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "rows": len(df),
        "date_range": {"start": str(df.index[0]), "end": str(df.index[-1])},
        "stats": {
            col: {"min": round(df[col].min(), 2), "max": round(df[col].max(), 2), "mean": round(df[col].mean(), 2)}
            for col in ["open", "high", "low", "close"]
            if col in df.columns
        },
        "last_5_rows": json.loads(df.tail(5).reset_index().to_json(orient="records", date_format="iso")),
    }
    return json.dumps(result, default=str)


@mcp.tool()
@log_tool_call
async def load_csv(file_path: str) -> str:
    """Load OHLCV data from a local CSV file.

    Args:
        file_path: Absolute or relative path to the CSV file

    Returns:
        JSON with row count, columns, date range, and first 5 rows.
    """
    df = _load_df(csv_path=file_path)
    result = {
        "file": file_path,
        "rows": len(df),
        "columns": list(df.columns),
        "date_range": {"start": str(df.index[0]), "end": str(df.index[-1])},
        "first_5_rows": json.loads(df.head(5).reset_index().to_json(orient="records", date_format="iso")),
    }
    return json.dumps(result, default=str)


# ===================================================================
# PATTERN TOOLS
# ===================================================================

@mcp.tool()
@log_tool_call
async def scan_all_patterns(
    symbol: str | None = None,
    timeframe: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    csv_path: str | None = None,
) -> str:
    """Scan price data for ALL chart patterns (double bottom/top, head & shoulders, flags, triangles, cup & handle).

    Provide either (symbol + start_date + end_date) to fetch data, or csv_path to load from file.

    Args:
        symbol: Ticker symbol (e.g. RELIANCE.NS)
        timeframe: Candle timeframe (default 1d)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        csv_path: Path to CSV file (alternative to symbol)

    Returns:
        JSON list of detected patterns with type, direction, target, stop loss, confidence.
    """
    df = _load_df(symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date, csv_path=csv_path)
    matches = scan_patterns(df)
    return json.dumps([_pattern_match_to_dict(m) for m in matches], default=str)


@mcp.tool()
@log_tool_call
async def scan_specific_pattern(
    pattern_name: str,
    symbol: str | None = None,
    timeframe: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    csv_path: str | None = None,
) -> str:
    """Scan price data for a SPECIFIC chart pattern.

    Args:
        pattern_name: Pattern to scan for (double_bottom, double_top, head_shoulders, flag_pole, cup_handle, triangle)
        symbol: Ticker symbol (e.g. RELIANCE.NS)
        timeframe: Candle timeframe (default 1d)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        csv_path: Path to CSV file (alternative to symbol)

    Returns:
        JSON list of detected patterns.
    """
    if pattern_name not in DETECTOR_MAP:
        return json.dumps({"error": f"Unknown pattern: {pattern_name}", "available": list(DETECTOR_MAP.keys())})
    df = _load_df(symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date, csv_path=csv_path)
    matches = scan_patterns(df, [pattern_name])
    return json.dumps([_pattern_match_to_dict(m) for m in matches], default=str)


@mcp.tool()
@log_tool_call
async def find_support_resistance(
    symbol: str | None = None,
    timeframe: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    csv_path: str | None = None,
) -> str:
    """Find support and resistance price levels in the data.

    Provide either (symbol + start_date + end_date) or csv_path.

    Args:
        symbol: Ticker symbol
        timeframe: Candle timeframe (default 1d)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        csv_path: Path to CSV file

    Returns:
        JSON with support_levels and resistance_levels arrays.
    """
    df = _load_df(symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date, csv_path=csv_path)
    support = find_support_levels(df)
    resistance = find_resistance_levels(df)
    return json.dumps({
        "support_levels": [round(s, 2) for s in support],
        "resistance_levels": [round(r, 2) for r in resistance],
    })


@mcp.tool()
@log_tool_call
async def list_available_patterns() -> str:
    """List all available chart pattern detectors.

    Returns:
        JSON list of pattern names that can be used with scan_specific_pattern.
    """
    return json.dumps(list(DETECTOR_MAP.keys()))


@mcp.tool()
@log_tool_call
async def generate_pattern_report(
    symbol: str | None = None,
    timeframe: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    csv_path: str | None = None,
    pattern_names: str | None = None,
) -> str:
    """Scan for chart patterns, evaluate outcomes, display a formatted report, and save it to disk.

    Args:
        symbol: Ticker symbol (e.g. GOOGL, RELIANCE.NS)
        timeframe: Candle timeframe (default 1d)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        csv_path: Path to CSV file (alternative to symbol)
        pattern_names: Comma-separated pattern names to scan (e.g. "flag_pole,double_bottom"). If empty, scans all.

    Returns:
        Formatted text report with pattern details, outcomes, and summary statistics.
        Also saved to storage/reports/.
    """
    df = _load_df(symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date, csv_path=csv_path)
    if df.empty:
        return "ERROR: No data found."

    names_list = [n.strip() for n in pattern_names.split(",")] if pattern_names else None
    matches = scan_patterns(df, names_list)

    label = symbol or csv_path or "unknown"
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"  PATTERN ANALYSIS REPORT — {label.upper()}")
    lines.append("=" * 80)
    lines.append(f"  Date range : {df.index[0].date()} to {df.index[-1].date()}")
    lines.append(f"  Bars       : {len(df)}")
    lines.append(f"  Price range: ${df['close'].min():.2f} – ${df['close'].max():.2f}")
    scanned = ", ".join(names_list) if names_list else "all"
    lines.append(f"  Patterns   : {scanned}")
    lines.append(f"  Found      : {len(matches)} pattern(s)")
    lines.append("=" * 80)

    wins = 0
    losses = 0
    open_trades = 0
    type_stats: dict[str, dict[str, int]] = {}

    for idx, m in enumerate(matches, 1):
        entry_price = df.iloc[m.end_idx]["close"]

        # Evaluate outcome
        post = df.iloc[m.end_idx + 1:] if m.end_idx + 1 < len(df) else df.iloc[0:0]
        outcome = "Open / no resolution"
        outcome_date = ""

        if not post.empty:
            if m.direction == "bullish":
                target_hit = post[post["high"] >= m.target_price]
                stop_hit = post[post["low"] <= m.stop_loss]
            else:
                target_hit = post[post["low"] <= m.target_price]
                stop_hit = post[post["high"] >= m.stop_loss]

            t_first = target_hit.index[0] if not target_hit.empty else None
            s_first = stop_hit.index[0] if not stop_hit.empty else None

            if t_first and (not s_first or t_first <= s_first):
                days = (t_first - m.confirmation_timestamp).days
                outcome = f"TARGET HIT ({days}d)"
                outcome_date = str(t_first.date())
                wins += 1
            elif s_first and (not t_first or s_first < t_first):
                days = (s_first - m.confirmation_timestamp).days
                outcome = f"STOP HIT ({days}d)"
                outcome_date = str(s_first.date())
                losses += 1
            else:
                open_trades += 1
        else:
            open_trades += 1

        # Track per-type stats
        pt = m.pattern_type
        if pt not in type_stats:
            type_stats[pt] = {"wins": 0, "losses": 0, "open": 0}
        if "TARGET" in outcome:
            type_stats[pt]["wins"] += 1
        elif "STOP" in outcome:
            type_stats[pt]["losses"] += 1
        else:
            type_stats[pt]["open"] += 1

        lines.append("")
        lines.append(f"  Pattern #{idx}: {m.pattern_type.upper()}")
        lines.append(f"  Direction    : {m.direction}")
        lines.append(f"  Period       : {df.index[m.start_idx].date()} → {m.confirmation_timestamp.date()}")
        lines.append(f"  Entry price  : ${entry_price:.2f}")
        lines.append(f"  Target       : ${m.target_price:.2f}")
        lines.append(f"  Stop loss    : ${m.stop_loss:.2f}")
        lines.append(f"  Confidence   : {m.confidence:.0%}")
        lines.append(f"  Pole move    : {m.metadata.get('pole_move_pct', 0):.1f}%")
        lines.append(f"  Flag retrace : {m.metadata.get('flag_retrace_pct', 0):.1f}%")
        lines.append(f"  Outcome      : {outcome}  {outcome_date}")
        lines.append("-" * 80)

    # Summary
    total_resolved = wins + losses
    win_rate = (wins / total_resolved * 100) if total_resolved else 0

    lines.append("")
    lines.append("=" * 80)
    lines.append("  SUMMARY")
    lines.append("=" * 80)
    lines.append(f"  Total patterns : {len(matches)}")
    lines.append(f"  Target hit     : {wins}")
    lines.append(f"  Stop hit       : {losses}")
    lines.append(f"  Open / pending : {open_trades}")
    lines.append(f"  Win rate       : {win_rate:.0f}% ({wins}/{total_resolved} resolved)")
    lines.append("")

    if type_stats:
        lines.append(f"  {'Pattern':<20} {'Wins':>6} {'Losses':>8} {'Open':>6} {'Win %':>7}")
        lines.append(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*6} {'-'*7}")
        for pt, st in type_stats.items():
            res = st["wins"] + st["losses"]
            wr = (st["wins"] / res * 100) if res else 0
            lines.append(f"  {pt:<20} {st['wins']:>6} {st['losses']:>8} {st['open']:>6} {wr:>6.0f}%")

    lines.append("")
    lines.append("=" * 80)

    report_text = "\n".join(lines)

    # Save to disk
    reports_dir = Path(_PROJECT_ROOT) / "storage" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pattern_{label.replace('.', '_')}_{ts}.txt"
    report_path = reports_dir / filename
    with open(report_path, "w") as f:
        f.write(report_text)

    report_text += f"\n\n  Report saved to: {report_path}"
    return report_text


# ===================================================================
# BACKTESTING TOOLS
# ===================================================================

@mcp.tool()
@log_tool_call
async def run_backtest(
    strategy: str,
    symbol: str,
    timeframe: str = "1d",
    start_date: str = "2020-01-01",
    end_date: str = "2025-01-01",
    params: str | None = None,
) -> str:
    """Run a backtest for a strategy on a given symbol and time period.

    Args:
        strategy: Strategy name (e.g. sma_crossover, ema_crossover, rsi_strategy, macd_strategy, bollinger_strategy)
        symbol: Ticker symbol (e.g. RELIANCE.NS)
        timeframe: Candle timeframe (default 1d)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        params: Optional JSON string of strategy parameters (e.g. '{"fast_period": 10, "slow_period": 50}')

    Returns:
        JSON with performance metrics, trade count, top winners and losers.
    """
    tf = Timeframe(timeframe)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    strategy_params = json.loads(params) if params else None

    runner = BacktestRunner(data_manager=_data_manager)
    result = runner.run_single(strategy, symbol, tf, start_dt, end_dt, params=strategy_params)

    output = result.to_dict()
    # Add top winners/losers summary
    output["top_winners"] = [
        {"entry_time": str(t.entry_time), "exit_time": str(t.exit_time), "pnl": round(t.pnl, 2), "pnl_pct": round(t.pnl_pct, 2)}
        for t in result.top_winners(5)
    ]
    output["top_losers"] = [
        {"entry_time": str(t.entry_time), "exit_time": str(t.exit_time), "pnl": round(t.pnl, 2), "pnl_pct": round(t.pnl_pct, 2)}
        for t in result.top_losers(5)
    ]
    # Remove full trade list to keep response manageable
    output.pop("trades", None)
    return json.dumps(output, default=str)


@mcp.tool()
@log_tool_call
async def run_parameter_sweep(
    strategy: str,
    symbol: str,
    timeframe: str = "1d",
    start_date: str = "2020-01-01",
    end_date: str = "2025-01-01",
) -> str:
    """Run a parameter sweep for a strategy, testing all parameter combinations.

    Args:
        strategy: Strategy name
        symbol: Ticker symbol (e.g. RELIANCE.NS)
        timeframe: Candle timeframe (default 1d)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD

    Returns:
        JSON with top 10 parameter combinations ranked by Sharpe ratio.
    """
    tf = Timeframe(timeframe)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    runner = BacktestRunner(data_manager=_data_manager)
    results = runner.run_parameter_sweep(strategy, symbol, tf, start_dt, end_dt)

    top_10 = []
    for r in results[:10]:
        top_10.append({
            "params": r.params,
            "total_return_pct": round(r.metrics.total_return_pct, 2),
            "sharpe_ratio": round(r.metrics.sharpe_ratio, 2),
            "max_drawdown_pct": round(r.metrics.max_drawdown_pct, 2),
            "win_rate_pct": round(r.metrics.win_rate_pct, 2),
            "total_trades": r.metrics.total_trades,
            "profit_factor": round(r.metrics.profit_factor, 2),
        })
    return json.dumps({"strategy": strategy, "symbol": symbol, "top_results": top_10}, default=str)


@mcp.tool()
@log_tool_call
async def list_strategies_tool() -> str:
    """List all available trading strategies for backtesting.

    Returns:
        JSON list of strategy names.
    """
    return json.dumps(list_strategies())


# ===================================================================
# INDICATOR TOOLS
# ===================================================================

@mcp.tool()
@log_tool_call
async def compute_indicators(
    indicators: str,
    symbol: str | None = None,
    timeframe: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    csv_path: str | None = None,
) -> str:
    """Compute technical indicators on price data.

    Args:
        indicators: Comma-separated list of indicators (e.g. "sma_20,ema_12,rsi_14,macd,bollinger,atr_14,obv,vwap")
        symbol: Ticker symbol
        timeframe: Candle timeframe (default 1d)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        csv_path: Path to CSV file (alternative to symbol)

    Returns:
        JSON with last 20 rows of computed indicator values.
    """
    df = _load_df(symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date, csv_path=csv_path)
    indicator_list = [ind.strip() for ind in indicators.split(",")]

    added_columns: list[str] = []
    original_cols = set(df.columns)

    for ind_name in indicator_list:
        try:
            df = add_indicator(df, ind_name)
            new_cols = set(df.columns) - original_cols
            added_columns.extend(new_cols)
            original_cols = set(df.columns)
        except ValueError as e:
            return json.dumps({"error": str(e)})

    # Return last 20 rows of indicator columns + close price
    cols_to_show = ["close"] + sorted(set(added_columns))
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    tail = df[cols_to_show].tail(20)
    rows = json.loads(tail.reset_index().to_json(orient="records", date_format="iso"))
    return json.dumps({"indicators": sorted(set(added_columns)), "rows": len(df), "last_20": rows}, default=str)


# ===================================================================
# EXECUTION TOOLS
# ===================================================================

@mcp.tool()
@log_tool_call
async def get_positions() -> str:
    """Get current open positions from Zerodha Kite.

    Returns:
        JSON list of open positions with symbol, quantity, avg price, LTP, P&L.
    """
    from execution.position_tracker import PositionTracker
    tracker = PositionTracker()
    details = tracker.get_position_details()
    return json.dumps(details, default=str)


@mcp.tool()
@log_tool_call
async def get_pnl() -> str:
    """Get today's realized and unrealized P&L from Zerodha Kite.

    Returns:
        JSON with realized, unrealized, and total P&L.
    """
    from execution.kite_broker import KiteBroker
    broker = KiteBroker()
    pnl = broker.get_pnl()
    return json.dumps(pnl, default=str)


@mcp.tool()
@log_tool_call
async def place_order(
    symbol: str,
    action: str,
    quantity: int,
    order_type: str = "MARKET",
    price: float | None = None,
    exchange: str = "NSE",
    product: str = "MIS",
) -> str:
    """Place a trading order via Zerodha Kite after risk validation.

    Args:
        symbol: Trading symbol (e.g. RELIANCE)
        action: BUY or SELL
        quantity: Number of shares
        order_type: MARKET, LIMIT, SL, or SL-M (default MARKET)
        price: Limit price (required for LIMIT and SL orders)
        exchange: Exchange — NSE, BSE, or NFO (default NSE)
        product: Product type — MIS (intraday), CNC (delivery), NRML (F&O) (default MIS)

    Returns:
        JSON with order_id on success, or rejection reason on failure.
    """
    from execution.kite_broker import KiteBroker
    from execution.risk_manager import RiskManager
    from strategies.base import Signal, SignalType

    action_upper = action.upper()
    signal_type = SignalType.BUY if action_upper == "BUY" else SignalType.SELL

    signal = Signal(
        timestamp=pd.Timestamp.now(),
        signal_type=signal_type,
        symbol=symbol,
        price=price or 0.0,
        quantity=quantity,
    )

    broker = KiteBroker()
    risk_mgr = RiskManager(broker=broker)
    is_valid, reason = risk_mgr.validate(signal)

    if not is_valid:
        return json.dumps({"status": "rejected", "reason": reason})

    order_id = broker.place_order(
        tradingsymbol=symbol,
        exchange=exchange,
        transaction_type=action_upper,
        order_type=order_type,
        quantity=quantity,
        product=product,
        price=price,
    )
    return json.dumps({"status": "placed", "order_id": order_id})


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")

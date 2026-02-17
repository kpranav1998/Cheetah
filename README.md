# Cheetah - AI Trading Agent

An AI-powered trading assistant for the Indian stock market. Combines chart pattern detection, backtesting, technical indicators, and live execution via Zerodha Kite — all accessible through a conversational AI agent in your terminal.

## Architecture

```
Terminal (stdin/stdout)
    |
LangGraph ReAct Agent (agent/)
    |
LiteLLM (gpt-4o, claude-3, llama, etc.)
    |
MCP Tools (16 tools via stdio transport)
    |
MCP Server (mcp_server/server.py)
    |
Core Modules: patterns/ backtesting/ data/ indicators/ execution/
```

The agent uses the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) to expose all trading capabilities as tools. A [LangGraph](https://github.com/langchain-ai/langgraph) ReAct agent reasons over these tools using any LLM via [LiteLLM](https://github.com/BerriAI/litellm).

## Features

### Chart Pattern Detection
Scans OHLCV price data for classical chart patterns:
- Double Bottom / Double Top
- Head and Shoulders / Inverse Head and Shoulders
- Bull Flag / Bear Flag
- Ascending / Descending / Symmetric Triangle
- Cup and Handle

Each detection returns entry price, target, stop loss, confidence score, and pattern metadata.

### Backtesting Engine
Full event-driven backtesting with realistic commission and slippage modeling:

| Strategy | Type |
|---|---|
| `sma_crossover` | SMA fast/slow crossover |
| `ema_crossover` | EMA fast/slow crossover |
| `macd_strategy` | MACD signal line crossover |
| `rsi_strategy` | RSI overbought/oversold |
| `bollinger_strategy` | Bollinger Band breakout |
| `pattern_strategy` | Trade on detected chart patterns |
| `iron_condor` | Options: Iron Condor |
| `bull_call_spread` | Options: Bull Call Spread |
| `bear_put_spread` | Options: Bear Put Spread |
| `straddle` | Options: Long Straddle |
| `strangle` | Options: Long Strangle |
| `covered_call` | Options: Covered Call |
| `protective_put` | Options: Protective Put |

Metrics: total return, CAGR, Sharpe/Sortino ratio, max drawdown, win rate, profit factor, expectancy.

Parameter sweep finds optimal strategy parameters automatically.

### Technical Indicators
- **Moving Averages**: SMA, EMA, DEMA, TEMA
- **Oscillators**: RSI, Stochastic, CCI, Williams %R
- **Trend**: MACD, ADX, SuperTrend
- **Volatility**: Bollinger Bands, ATR, Keltner Channel
- **Volume**: OBV, VWAP, MFI

### Live Execution (Zerodha Kite)
- Place orders (Market, Limit, SL, SL-M)
- Track open positions and P&L
- Risk management: daily loss limits, max positions, position sizing

### Data
- Yahoo Finance for historical OHLCV data (NSE stocks via `.NS` suffix)
- Parquet cache with SQLite index for fast repeated access
- CSV file loading for custom datasets

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/kpranav1998/Cheetah.git
cd Cheetah
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# Required for the AI agent
LITELLM_MODEL=gpt-4o
LLM_API_KEY=your_openai_or_anthropic_key

# Optional: Zerodha Kite (for live trading only)
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret
KITE_ACCESS_TOKEN=your_access_token
```

### 3. Run the AI agent

```bash
python run_agent.py
```

Or specify a different model:

```bash
python run_agent.py --model gpt-4o
python run_agent.py --model claude-sonnet-4-20250514
python run_agent.py --model ollama/llama3
```

### 4. Chat with the agent

```
You> List all available trading strategies
You> Scan RELIANCE.NS for chart patterns from 2023 to 2025
You> Backtest SMA crossover on TCS.NS daily from 2020 to 2025
You> Run parameter sweep for RSI strategy on INFY.NS from 2020 to 2024
You> Compute RSI, MACD, and Bollinger Bands for RELIANCE.NS last 2 years
You> Find support and resistance levels for ^NSEI from 2023 to 2025
```

## CLI (without the AI agent)

The project also includes a traditional CLI via [Typer](https://typer.tiangolo.com/):

```bash
# Backtest a strategy
python main.py backtest --strategy sma_crossover --symbol RELIANCE.NS --start 2020-01-01 --end 2025-01-01

# Parameter sweep
python main.py sweep --strategy ema_crossover --symbol TCS.NS --start 2020-01-01 --end 2025-01-01

# Scan CSV for patterns
python main.py scan sample_reliance.csv
python main.py scan sample_reliance.csv --pattern flag_pole

# List strategies
python main.py list

# View live positions (requires Kite credentials)
python main.py positions

# Backtest + LLM analysis
python main.py analyze --strategy sma_crossover --symbol RELIANCE.NS --start 2020-01-01 --end 2025-01-01 --report
```

## MCP Server (standalone)

The MCP server can be used independently with any MCP client:

```bash
python -m mcp_server.server
```

This starts an MCP server on stdio transport exposing 16 tools:

| Category | Tools |
|---|---|
| **Data** | `fetch_ohlcv`, `load_csv` |
| **Patterns** | `scan_all_patterns`, `scan_specific_pattern`, `find_support_resistance`, `list_available_patterns` |
| **Backtesting** | `run_backtest`, `run_parameter_sweep`, `list_strategies_tool` |
| **Indicators** | `compute_indicators` |
| **Execution** | `get_positions`, `get_pnl`, `place_order` |

## Project Structure

```
Cheetah/
├── run_agent.py              # Entry point for AI agent
├── main.py                   # CLI entry point (Typer)
├── requirements.txt
├── .env.example
│
├── agent/                    # LangGraph ReAct agent
│   ├── chat.py               # Terminal chat loop + MCP client
│   ├── graph.py              # StateGraph definition
│   └── state.py              # AgentState TypedDict
│
├── mcp_server/               # MCP server (stdio transport)
│   └── server.py             # 16 tool definitions wrapping core modules
│
├── patterns/                 # Chart pattern detection
│   ├── scanner.py            # scan_patterns() entry point
│   ├── base.py               # PatternMatch dataclass
│   ├── support_resistance.py # Support/resistance level detection
│   ├── double_bottom.py
│   ├── double_top.py
│   ├── head_shoulders.py
│   ├── flag_pole.py
│   ├── cup_handle.py
│   └── triangles.py
│
├── backtesting/              # Backtesting engine
│   ├── runner.py             # BacktestRunner (single, sweep, multi-symbol)
│   ├── engine.py             # BacktestEngine + BacktestConfig
│   ├── options_engine.py     # Options backtesting
│   ├── metrics.py            # Performance metrics computation
│   ├── portfolio.py          # Position + Portfolio tracking
│   ├── result.py             # BacktestResult with serialization
│   └── trade.py              # Trade dataclass
│
├── strategies/               # Trading strategies
│   ├── base.py               # BaseStrategy ABC, Signal, SignalType
│   ├── registry.py           # @register decorator, get_strategy()
│   ├── equity/               # Equity strategies (SMA, EMA, MACD, RSI, Bollinger, Pattern)
│   └── options/              # Options strategies (Iron Condor, Spreads, Straddle, etc.)
│
├── indicators/               # Technical indicators
│   ├── __init__.py           # add_indicator() dispatcher
│   ├── moving_averages.py    # SMA, EMA, DEMA, TEMA
│   ├── oscillators.py        # RSI, Stochastic, CCI, Williams %R
│   ├── trend.py              # MACD, ADX, SuperTrend
│   ├── volatility.py         # Bollinger Bands, ATR, Keltner Channel
│   └── volume.py             # OBV, VWAP, MFI
│
├── data/                     # Data access layer
│   ├── data_manager.py       # DataManager (cache-first, then Yahoo)
│   ├── cache.py              # ParquetCache with SQLite index
│   ├── fetcher_yahoo.py      # Yahoo Finance fetcher
│   ├── fetcher_kite.py       # Kite Connect fetcher
│   └── options_chain.py      # Options chain fetcher
│
├── execution/                # Live trading via Zerodha Kite
│   ├── kite_broker.py        # KiteBroker (orders, positions, P&L)
│   ├── order_manager.py      # Signal-to-order conversion
│   ├── position_tracker.py   # Position tracking
│   └── risk_manager.py       # Risk validation + position sizing
│
├── analysis/                 # LLM-powered analysis
│   ├── llm_client.py         # OpenAI client wrapper
│   ├── result_analyzer.py    # Backtest result analysis
│   ├── report_generator.py   # HTML report generation
│   └── parameter_optimizer.py
│
├── options_pricing/          # Options pricing models
│   ├── black_scholes.py
│   ├── implied_vol.py
│   └── payoff.py
│
├── config/                   # Configuration
│   ├── settings.py           # AppSettings (pydantic-settings)
│   ├── timeframes.py         # Timeframe enum
│   └── instruments.py        # NSE lot sizes and charges
│
├── utils/                    # Utilities
│   ├── logger.py             # Structured logging (JSON/text, rotation, correlation IDs)
│   └── nse_calendar.py       # Trading day/expiry calculations
│
├── storage/                  # Runtime data (gitignored)
│   ├── cache/                # Parquet data cache
│   ├── results/              # Backtest result JSONs
│   ├── reports/              # HTML reports
│   └── logs/                 # Rotating log files
│
└── tests/                    # Test suite
```

## Configuration

All settings are managed via environment variables (`.env` file) and `config/settings.py`:

| Variable | Default | Description |
|---|---|---|
| `LITELLM_MODEL` | `gpt-4o` | LLM model for the agent (any LiteLLM-supported model) |
| `AGENT_TEMPERATURE` | `0.1` | LLM temperature |
| `LLM_API_KEY` | — | API key for the LLM provider |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FORMAT` | `text` | Log format (`text` for dev, `json` for production) |
| `LOG_FILE` | auto | Log file path (default: `storage/logs/trading_agent.log`) |
| `KITE_API_KEY` | — | Zerodha API key (live trading only) |
| `KITE_API_SECRET` | — | Zerodha API secret |
| `KITE_ACCESS_TOKEN` | — | Zerodha session token |

## Logging

Production-grade structured logging with:
- **JSON format** for production: `{"timestamp", "level", "logger", "message", "request_id", "duration_ms"}`
- **Text format** for development: `2025-01-15 10:30:00 [module] INFO: message | key=value`
- **Rotating file handler**: 10MB max, 5 backups
- **Correlation IDs**: Each conversation turn gets a unique `request_id`
- **Tool call logging**: `@log_tool_call` decorator auto-logs invocation, duration, and errors

## Supported Symbols

- **NSE Stocks**: Append `.NS` suffix — `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`
- **Nifty 50 Index**: `^NSEI`
- **Bank Nifty**: `^NSEBANK`
- **Any Yahoo Finance symbol**: Works with US stocks too (`AAPL`, `GOOGL`, etc.)

## License

MIT

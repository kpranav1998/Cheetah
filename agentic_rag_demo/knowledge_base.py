"""
knowledge_base.py
-----------------
8 short documents about trading concepts.
These form the knowledge base the RAG agent will query over.
"""

DOCUMENTS = [
    {
        "id": "doc_1",
        "title": "Moving Averages",
        "content": """
Moving averages are one of the most widely used technical indicators in trading.
They smooth out price data by creating a constantly updated average price over a
specific time period.

A Simple Moving Average (SMA) calculates the arithmetic mean of a set of prices
over a specified number of periods. For example, a 10-day SMA adds up the closing
prices of the last 10 days and divides by 10. Every day the oldest price is dropped
and the newest is added. SMA gives equal weight to every data point in the window.

An Exponential Moving Average (EMA) gives more weight to recent prices, making it
more responsive to new information. The weighting factor decreases exponentially for
older data points. EMA reacts faster to price changes than SMA, which makes it
preferred for short-term trading. The formula uses a smoothing factor k = 2/(n+1)
where n is the number of periods.

A Weighted Moving Average (WMA) assigns linearly increasing weights to more recent
data. Unlike EMA, WMA weights decrease linearly rather than exponentially.

Key uses of moving averages:
- Trend identification: price above SMA suggests uptrend, below suggests downtrend
- Support and resistance levels
- Crossover signals: when a fast MA crosses a slow MA, it signals a trend change
- The Golden Cross (50-day SMA crossing above 200-day SMA) is a bullish signal
- The Death Cross (50-day SMA crossing below 200-day SMA) is a bearish signal

Common periods: 9, 20, 50, 100, 200 days for daily charts; 9, 21 for intraday.
Moving averages are lagging indicators — they confirm trends rather than predict them.
        """.strip(),
    },
    {
        "id": "doc_2",
        "title": "RSI - Relative Strength Index",
        "content": """
The Relative Strength Index (RSI) is a momentum oscillator developed by J. Welles
Wilder in 1978. It measures the speed and magnitude of price changes to evaluate
overbought or oversold conditions.

RSI is calculated over a default period of 14 days. The formula compares the average
gain on up days to the average loss on down days over the look-back period:
RSI = 100 - (100 / (1 + RS)) where RS = Average Gain / Average Loss.

RSI oscillates between 0 and 100. Traditionally:
- RSI above 70: asset is overbought — a potential sell signal
- RSI below 30: asset is oversold — a potential buy signal
- RSI of 50 represents a neutral momentum

However, in strong trends, RSI can stay in overbought or oversold territory for
extended periods. In a strong uptrend, RSI often stays between 40-90. In a strong
downtrend, RSI often stays between 10-60. This is called RSI range shift.

RSI divergence is a powerful signal:
- Bullish divergence: price makes a lower low but RSI makes a higher low — suggests
  weakening downward momentum and potential reversal upward
- Bearish divergence: price makes a higher high but RSI makes a lower high — suggests
  weakening upward momentum and potential reversal downward

RSI can also be used to identify trend: RSI consistently above 50 confirms an uptrend,
consistently below 50 confirms a downtrend.

Common RSI strategies:
1. Classic overbought/oversold: buy below 30, sell above 70
2. RSI divergence: look for price-RSI divergence at key levels
3. RSI trendline breaks: draw trendlines on RSI itself
4. Multi-timeframe RSI: combine RSI signals from different timeframes for confirmation
        """.strip(),
    },
    {
        "id": "doc_3",
        "title": "MACD Indicator",
        "content": """
The Moving Average Convergence Divergence (MACD) is a trend-following momentum
indicator that shows the relationship between two exponential moving averages.

The MACD is calculated as:
MACD Line = 12-period EMA minus 26-period EMA
Signal Line = 9-period EMA of the MACD Line
Histogram = MACD Line minus Signal Line

When the MACD line crosses above the signal line, it generates a bullish crossover
(buy signal). When it crosses below, it generates a bearish crossover (sell signal).

The MACD histogram visualizes the distance between MACD and signal lines. When
histogram bars grow, momentum is increasing. When they shrink, momentum is fading.

MACD signals:
1. Signal line crossover: most common signal; bullish when MACD crosses above signal
2. Zero line crossover: when MACD crosses above zero, uptrend is confirmed
3. MACD divergence: similar to RSI divergence — powerful reversal signal
4. Histogram divergence: histogram making lower highs while price makes higher highs

MACD strengths: works well in trending markets, easy to read visually.
MACD weaknesses: it is a lagging indicator, prone to false signals in ranging markets.

The standard parameters (12, 26, 9) were optimized for daily stock charts. For
different timeframes or asset classes, parameters may need adjustment. Faster settings
like (5, 13, 4) are used for short-term trading; slower settings for longer trends.

MACD is most powerful when combined with other indicators — use RSI to confirm
momentum and volume to confirm breakouts alongside MACD signals.
        """.strip(),
    },
    {
        "id": "doc_4",
        "title": "Momentum Trading Strategy",
        "content": """
Momentum trading is a strategy based on the idea that assets that have performed
well recently tend to continue performing well in the near future, and assets that
have performed poorly tend to continue underperforming.

The core principle is: buy high, sell higher. Momentum traders look for assets
showing strong directional movement and ride the trend until it shows signs of
reversing.

Key concepts in momentum trading:

Rate of Change (ROC): measures the percentage change in price over a specific period.
High ROC values indicate strong momentum. Traders buy assets with highest recent ROC.

Relative Strength (not RSI): comparing one asset's performance against a benchmark
or other assets. Assets outperforming peers show relative momentum.

Momentum Entry Signals:
- Price breaking above a key resistance level with high volume
- New 52-week highs or all-time highs (breakout momentum)
- EMA crossovers in the direction of the trend
- RSI crossing above 50 from below (momentum shift confirmation)

Momentum Exit Signals:
- Price falling below a key moving average (e.g., 20-day EMA)
- RSI entering overbought territory and reversing (above 70 then declining)
- Volume declining while price is rising (weak momentum)
- MACD histogram showing diminishing bars

Risk management in momentum trading is critical. Momentum stocks can reverse sharply.
Use stop-losses at meaningful technical levels (below key MA or recent swing low).
Position sizing should account for the higher volatility of momentum names.

Momentum strategies work best in trending markets and can suffer during mean-reverting
or choppy market conditions. Common lookback periods: 3, 6, 12 months for position
trading; 5-20 days for swing trading.
        """.strip(),
    },
    {
        "id": "doc_5",
        "title": "Mean Reversion Strategy",
        "content": """
Mean reversion is a trading strategy based on the statistical tendency of prices
to return to their historical average after deviating significantly from it.

The core assumption: prices fluctuate around a long-term mean and extreme deviations
are temporary. When a price moves far from its mean, it is more likely to reverse
back toward the average than to continue moving further away.

Key tools for mean reversion:

Bollinger Bands: consist of a middle band (20-day SMA) plus upper and lower bands
at 2 standard deviations. When price touches the upper band, it may be overbought
(sell signal); when it touches the lower band, it may be oversold (buy signal).
The "squeeze" (narrow bands) signals low volatility often preceding a breakout.

Z-Score: measures how many standard deviations a price is from its mean.
Z = (Current Price - Mean) / Standard Deviation
A z-score above 2 signals potential short; below -2 signals potential long.

Pairs Trading: a market-neutral strategy where you trade two correlated assets.
When their price ratio or spread diverges significantly, you go long the underperformer
and short the outperformer, expecting the spread to converge.

RSI in mean reversion: RSI below 30 = oversold = buy; RSI above 70 = overbought = sell.
This is the classic mean-reversion interpretation of RSI.

Mean reversion strategies work best in:
- Range-bound, sideways markets
- Highly liquid assets (indices, large-cap stocks, forex majors)
- On shorter timeframes where noise dominates

They struggle in strongly trending markets where prices can remain extended for long.
Key risk: a trending move can wipe out a mean-reversion trader who keeps averaging
against the trend. Always use stop-losses and avoid over-leveraging.
        """.strip(),
    },
    {
        "id": "doc_6",
        "title": "Risk Management",
        "content": """
Risk management is the most critical aspect of successful trading. Even a strategy
with a 40% win rate can be profitable with proper risk management, while a 70% win
rate strategy can be ruinous with poor risk management.

Position Sizing - The 1% Rule:
Never risk more than 1-2% of your total capital on a single trade. If you have
Rs. 1,00,000 capital, your maximum loss per trade should be Rs. 1,000-2,000.
Position size = (Capital × Risk%) / (Entry Price - Stop Loss Price)

Stop Loss Orders:
A stop loss is a pre-defined price at which you will exit a losing trade. Types:
- Fixed stop: set at a percentage below entry (e.g., 2% below buy price)
- Technical stop: set below a key support level or moving average
- Volatility stop: uses ATR (Average True Range) to set stops based on volatility
- Trailing stop: moves with the price, locking in profits as price rises

Risk-to-Reward Ratio (R:R):
Only enter trades where potential profit is at least 2x the potential loss (2:1 R:R).
With a 2:1 R:R, you only need to be right 34% of the time to break even.
Aim for 3:1 or higher for swing trades.

Maximum Drawdown:
Track your maximum drawdown (peak-to-trough decline). If drawdown exceeds 20%,
reduce position sizes. Most professional funds have hard stops at 25-30% drawdown.

Diversification:
Do not put all capital in one sector or asset class. Diversify across:
- Asset classes (equities, bonds, commodities)
- Sectors (tech, pharma, FMCG, banking)
- Timeframes (intraday, swing, position)

Avoid over-trading, revenge trading after losses, and averaging down into losing
positions. Keep a trading journal to track performance and identify patterns in losses.
        """.strip(),
    },
    {
        "id": "doc_7",
        "title": "Backtesting",
        "content": """
Backtesting is the process of testing a trading strategy on historical data to
evaluate how it would have performed in the past. It is a crucial step before
deploying any trading strategy with real capital.

Steps in backtesting:
1. Define the strategy rules precisely (entry, exit, position sizing, risk management)
2. Gather clean historical data for the assets to be traded
3. Implement the strategy logic (in code or using software)
4. Run the simulation over the historical period
5. Analyze the performance metrics

Key backtesting metrics:
- Total Return: overall percentage gain/loss over the period
- CAGR: Compound Annual Growth Rate — annualized return
- Sharpe Ratio: (Return - Risk-Free Rate) / Standard Deviation of Returns
  A Sharpe above 1.0 is good; above 2.0 is excellent
- Max Drawdown: largest peak-to-trough decline — measures worst-case scenario
- Win Rate: percentage of trades that were profitable
- Profit Factor: Gross Profit / Gross Loss — above 1.5 is solid
- Average Win / Average Loss ratio

Common backtesting pitfalls:
1. Overfitting (curve fitting): optimizing parameters too closely to historical data.
   The strategy performs great historically but fails in live trading.
2. Look-ahead bias: accidentally using future data in strategy signals
3. Survivorship bias: only testing on stocks that survived (ignores failed companies)
4. Transaction costs: not accounting for brokerage, slippage, and impact costs
5. Data snooping bias: testing many strategies on the same data inflates apparent performance

Best practices:
- Use walk-forward testing: optimize on in-sample data, validate on out-of-sample data
- Always validate on data the strategy has never seen
- Test across multiple market regimes (bull, bear, sideways)
- Paper trade before going live
        """.strip(),
    },
    {
        "id": "doc_8",
        "title": "Order Types and Market Microstructure",
        "content": """
Understanding order types is fundamental for executing trading strategies effectively.
The wrong order type can significantly impact profitability through slippage and
unfavorable fills.

Market Order:
Executes immediately at the best available price. Guarantees execution but not price.
Use when: you need to enter/exit urgently and price certainty is less important.
Risk: in illiquid markets or during news events, can get very poor fills (slippage).

Limit Order:
Executes only at the specified price or better. Guarantees price but not execution.
Buy limit: placed below current price (you want to buy cheaper)
Sell limit: placed above current price (you want to sell at a better price)
Use when: entering positions at desired prices or taking profit at target levels.

Stop Order (Stop Loss):
Becomes a market order when price reaches the stop price. Used to limit losses.
Stop-limit order: becomes a limit order at the stop price — safer but may not fill
during fast markets.

Bracket Order: combines entry, stop loss, and target in a single order. Common in
intraday trading for automated risk management.

Slippage: the difference between the expected price and the actual fill price.
Causes: market impact (your order moves the price), liquidity gaps, fast markets.
Impact: slippage can turn a profitable backtest strategy into a losing live strategy.
Always include slippage estimates in backtesting.

Bid-Ask Spread: the difference between the best buy price (bid) and best sell price (ask).
You buy at ask and sell at bid — the spread is the immediate cost of a round-trip trade.
Liquid assets have narrow spreads (e.g., Nifty futures: 0.05 points).
Illiquid assets have wide spreads that erode profits.

Order Book: a real-time list of buy and sell orders at different price levels.
Reading the order book (Level 2 data) gives insight into near-term supply and demand.
        """.strip(),
    },
]

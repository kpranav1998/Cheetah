from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf

from config.timeframes import Timeframe


class YahooFetcher:
    """Fetches historical OHLCV data from Yahoo Finance."""

    # yfinance limits intraday data to last 60 days for 1m,
    # last 730 days for 1h, unlimited for daily/weekly.
    MAX_INTRADAY_DAYS = {
        Timeframe.MIN_1: 7,
        Timeframe.MIN_5: 60,
        Timeframe.MIN_15: 60,
        Timeframe.MIN_30: 60,
        Timeframe.HOUR_1: 730,
    }

    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data. Returns DataFrame with columns:
        open, high, low, close, volume (lowercase) and datetime index.

        For NSE symbols, append .NS (e.g., RELIANCE.NS, ^NSEI for Nifty 50).
        """
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=timeframe.yfinance_interval,
            auto_adjust=True,
        )

        if df.empty:
            return df

        df.columns = [c.lower() for c in df.columns]
        # Keep only OHLCV columns
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]
        df.index.name = "datetime"
        df.attrs["symbol"] = symbol
        df.attrs["timeframe"] = timeframe.value
        return df

    def fetch_multiple(
        self,
        symbols: list[str],
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> dict[str, pd.DataFrame]:
        results = {}
        for symbol in symbols:
            results[symbol] = self.fetch(symbol, timeframe, start, end)
        return results

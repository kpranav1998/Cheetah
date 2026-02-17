from __future__ import annotations

from datetime import datetime

import pandas as pd

from config.settings import settings
from config.timeframes import Timeframe
from data.cache import ParquetCache
from data.fetcher_yahoo import YahooFetcher


class DataManager:
    """Unified data access. Cache-first, then fetch from remote."""

    def __init__(self):
        self.cache = ParquetCache(settings.cache_dir)
        self.yahoo = YahooFetcher()

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        source: str = "auto",
    ) -> pd.DataFrame:
        # Try cache first
        if self.cache.has_range(symbol, timeframe, start, end):
            cached = self.cache.get(symbol, timeframe, start, end)
            if cached is not None and not cached.empty:
                return cached

        # Fetch from Yahoo
        df = self.yahoo.fetch(symbol, timeframe, start, end)

        if not df.empty:
            self.cache.store(symbol, timeframe, df)

        df.attrs["symbol"] = symbol
        df.attrs["timeframe"] = timeframe.value
        return df

    def get_multiple(
        self,
        symbols: list[str],
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> dict[str, pd.DataFrame]:
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_ohlcv(symbol, timeframe, start, end)
        return results

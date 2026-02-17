from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from kiteconnect import KiteConnect

from config.settings import settings
from config.timeframes import Timeframe


KITE_INTERVAL_MAP = {
    Timeframe.MIN_1: "minute",
    Timeframe.MIN_5: "5minute",
    Timeframe.MIN_15: "15minute",
    Timeframe.MIN_30: "30minute",
    Timeframe.HOUR_1: "60minute",
    Timeframe.DAILY: "day",
}


class KiteFetcher:
    """Fetch historical and live data from Zerodha Kite Connect."""

    def __init__(self):
        self.kite = KiteConnect(api_key=settings.kite_api_key)
        if settings.kite_access_token:
            self.kite.set_access_token(settings.kite_access_token)

    def set_access_token(self, token: str) -> None:
        self.kite.set_access_token(token)
        settings.kite_access_token = token

    def login_url(self) -> str:
        return self.kite.login_url()

    def generate_session(self, request_token: str) -> dict:
        data = self.kite.generate_session(request_token, api_secret=settings.kite_api_secret)
        self.set_access_token(data["access_token"])
        return data

    def fetch_historical(
        self,
        instrument_token: int,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch historical candle data from Kite."""
        interval = KITE_INTERVAL_MAP.get(timeframe, "day")

        # Kite limits: 60 days for minute data, 400 days for daily
        all_data = []
        current_start = start

        while current_start < end:
            if timeframe in (Timeframe.MIN_1, Timeframe.MIN_5, Timeframe.MIN_15, Timeframe.MIN_30):
                chunk_end = min(current_start + timedelta(days=60), end)
            else:
                chunk_end = min(current_start + timedelta(days=400), end)

            data = self.kite.historical_data(
                instrument_token,
                current_start,
                chunk_end,
                interval,
            )
            all_data.extend(data)
            current_start = chunk_end + timedelta(days=1)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.columns = [c.lower() for c in df.columns]
        df.index.name = "datetime"
        return df[["open", "high", "low", "close", "volume"]]

    def get_instruments(self, exchange: str = "NSE") -> pd.DataFrame:
        instruments = self.kite.instruments(exchange)
        return pd.DataFrame(instruments)

    def get_ltp(self, symbols: list[str]) -> dict[str, float]:
        """Get last traded price. symbols format: 'NSE:RELIANCE'"""
        data = self.kite.ltp(symbols)
        return {s: d["last_price"] for s, d in data.items()}

    def get_quote(self, symbols: list[str]) -> dict:
        return self.kite.quote(symbols)

"""Indicator dispatcher - adds indicators to a DataFrame by name."""

import pandas as pd

from indicators.moving_averages import add_sma, add_ema
from indicators.oscillators import add_rsi
from indicators.trend import add_macd
from indicators.volatility import add_bollinger, add_atr
from indicators.volume import add_obv, add_vwap


def add_indicator(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Add a named indicator to the DataFrame. E.g., 'sma_20', 'rsi_14', 'macd'."""
    if name.startswith("sma_"):
        period = int(name.split("_")[1])
        return add_sma(df, period)
    elif name.startswith("ema_"):
        period = int(name.split("_")[1])
        return add_ema(df, period)
    elif name.startswith("rsi"):
        period = int(name.split("_")[1]) if "_" in name else 14
        return add_rsi(df, period)
    elif name == "macd":
        return add_macd(df)
    elif name.startswith("bb") or name == "bollinger":
        return add_bollinger(df)
    elif name.startswith("atr"):
        period = int(name.split("_")[1]) if "_" in name else 14
        return add_atr(df, period)
    elif name == "obv":
        return add_obv(df)
    elif name == "vwap":
        return add_vwap(df)
    else:
        raise ValueError(f"Unknown indicator: {name}")

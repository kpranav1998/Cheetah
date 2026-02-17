import pandas as pd
import numpy as np


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    direction = typical_price.diff()
    pos_flow = raw_money_flow.where(direction > 0, 0.0)
    neg_flow = raw_money_flow.where(direction < 0, 0.0)
    pos_sum = pos_flow.rolling(window=period).sum()
    neg_sum = neg_flow.rolling(window=period).sum()
    mfi_ratio = pos_sum / neg_sum
    return 100 - (100 / (1 + mfi_ratio))


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    df["obv"] = obv(df["close"], df["volume"])
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df["vwap"] = vwap(df["high"], df["low"], df["close"], df["volume"])
    return df

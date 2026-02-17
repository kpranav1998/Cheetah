import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def dema(series: pd.Series, period: int) -> pd.Series:
    e = ema(series, period)
    return 2 * e - ema(e, period)


def tema(series: pd.Series, period: int) -> pd.Series:
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3 * e1 - 3 * e2 + e3


def add_sma(df: pd.DataFrame, period: int, column: str = "close") -> pd.DataFrame:
    df[f"sma_{period}"] = sma(df[column], period)
    return df


def add_ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.DataFrame:
    df[f"ema_{period}"] = ema(df[column], period)
    return df

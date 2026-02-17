from __future__ import annotations

import numpy as np
import pandas as pd


def find_pivots(
    df: pd.DataFrame, left: int = 5, right: int = 5
) -> tuple[pd.Series, pd.Series]:
    """Find pivot highs and pivot lows using left/right bar comparison."""
    highs = pd.Series(np.nan, index=df.index)
    lows = pd.Series(np.nan, index=df.index)

    for i in range(left, len(df) - right):
        # Pivot high: higher than all left and right neighbors
        window_high = df["high"].iloc[i - left : i + right + 1]
        if df["high"].iloc[i] == window_high.max():
            highs.iloc[i] = df["high"].iloc[i]

        # Pivot low: lower than all left and right neighbors
        window_low = df["low"].iloc[i - left : i + right + 1]
        if df["low"].iloc[i] == window_low.min():
            lows.iloc[i] = df["low"].iloc[i]

    return highs, lows


def find_support_levels(
    df: pd.DataFrame, left: int = 5, right: int = 5, tolerance_pct: float = 1.0
) -> list[float]:
    """Find support levels by clustering pivot lows."""
    _, pivot_lows = find_pivots(df, left, right)
    lows = pivot_lows.dropna().values
    return _cluster_levels(lows, tolerance_pct)


def find_resistance_levels(
    df: pd.DataFrame, left: int = 5, right: int = 5, tolerance_pct: float = 1.0
) -> list[float]:
    """Find resistance levels by clustering pivot highs."""
    pivot_highs, _ = find_pivots(df, left, right)
    highs = pivot_highs.dropna().values
    return _cluster_levels(highs, tolerance_pct)


def _cluster_levels(values: np.ndarray, tolerance_pct: float) -> list[float]:
    """Cluster nearby price levels together."""
    if len(values) == 0:
        return []

    sorted_vals = np.sort(values)
    clusters: list[list[float]] = [[sorted_vals[0]]]

    for val in sorted_vals[1:]:
        if abs(val - np.mean(clusters[-1])) / np.mean(clusters[-1]) * 100 < tolerance_pct:
            clusters[-1].append(val)
        else:
            clusters.append([val])

    # Return mean of each cluster, weighted by number of touches
    return [np.mean(c) for c in clusters if len(c) >= 2]

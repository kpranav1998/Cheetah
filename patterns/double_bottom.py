from __future__ import annotations

import numpy as np
import pandas as pd

from patterns.base import BasePatternDetector, PatternMatch
from patterns.support_resistance import find_pivots


class DoubleBottomDetector(BasePatternDetector):
    name = "double_bottom"

    def __init__(self, tolerance_pct: float = 2.0, min_gap_bars: int = 10, max_gap_bars: int = 60):
        self.tolerance_pct = tolerance_pct
        self.min_gap_bars = min_gap_bars
        self.max_gap_bars = max_gap_bars

    def detect(self, df: pd.DataFrame) -> list[PatternMatch]:
        _, pivot_lows = find_pivots(df, left=5, right=5)
        low_indices = pivot_lows.dropna().index.tolist()
        matches = []

        for i in range(len(low_indices)):
            for j in range(i + 1, len(low_indices)):
                idx_a = df.index.get_loc(low_indices[i])
                idx_b = df.index.get_loc(low_indices[j])
                gap = idx_b - idx_a

                if gap < self.min_gap_bars or gap > self.max_gap_bars:
                    continue

                price_a = pivot_lows[low_indices[i]]
                price_b = pivot_lows[low_indices[j]]

                # Two lows should be at similar levels
                pct_diff = abs(price_a - price_b) / price_a * 100
                if pct_diff > self.tolerance_pct:
                    continue

                # Neckline: highest point between the two lows
                between = df.iloc[idx_a:idx_b + 1]
                neckline = between["high"].max()

                # Confirm: price breaks above neckline after second bottom
                confirm_start = idx_b + 1
                if confirm_start >= len(df):
                    continue

                remaining = df.iloc[confirm_start:min(confirm_start + 20, len(df))]
                breakout = remaining[remaining["close"] > neckline]

                if breakout.empty:
                    continue

                confirm_ts = breakout.index[0]
                # Measured move target = neckline + (neckline - bottom)
                bottom = min(price_a, price_b)
                target = neckline + (neckline - bottom)

                matches.append(PatternMatch(
                    pattern_type="double_bottom",
                    start_idx=idx_a,
                    end_idx=df.index.get_loc(confirm_ts),
                    confirmation_timestamp=confirm_ts,
                    direction="bullish",
                    target_price=target,
                    stop_loss=bottom * 0.99,
                    confidence=min(1.0, 0.5 + (1.0 - pct_diff / self.tolerance_pct) * 0.5),
                ))

        return matches

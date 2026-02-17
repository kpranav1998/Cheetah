from __future__ import annotations

import numpy as np
import pandas as pd

from patterns.base import BasePatternDetector, PatternMatch
from patterns.support_resistance import find_pivots


class DoubleTopDetector(BasePatternDetector):
    name = "double_top"

    def __init__(self, tolerance_pct: float = 2.0, min_gap_bars: int = 10, max_gap_bars: int = 60):
        self.tolerance_pct = tolerance_pct
        self.min_gap_bars = min_gap_bars
        self.max_gap_bars = max_gap_bars

    def detect(self, df: pd.DataFrame) -> list[PatternMatch]:
        pivot_highs, _ = find_pivots(df, left=5, right=5)
        high_indices = pivot_highs.dropna().index.tolist()
        matches = []

        for i in range(len(high_indices)):
            for j in range(i + 1, len(high_indices)):
                idx_a = df.index.get_loc(high_indices[i])
                idx_b = df.index.get_loc(high_indices[j])
                gap = idx_b - idx_a

                if gap < self.min_gap_bars or gap > self.max_gap_bars:
                    continue

                price_a = pivot_highs[high_indices[i]]
                price_b = pivot_highs[high_indices[j]]

                pct_diff = abs(price_a - price_b) / price_a * 100
                if pct_diff > self.tolerance_pct:
                    continue

                # Neckline: lowest point between two tops
                between = df.iloc[idx_a:idx_b + 1]
                neckline = between["low"].min()

                # Confirm: price breaks below neckline
                confirm_start = idx_b + 1
                if confirm_start >= len(df):
                    continue

                remaining = df.iloc[confirm_start:min(confirm_start + 20, len(df))]
                breakout = remaining[remaining["close"] < neckline]

                if breakout.empty:
                    continue

                confirm_ts = breakout.index[0]
                top = max(price_a, price_b)
                target = neckline - (top - neckline)

                matches.append(PatternMatch(
                    pattern_type="double_top",
                    start_idx=idx_a,
                    end_idx=df.index.get_loc(confirm_ts),
                    confirmation_timestamp=confirm_ts,
                    direction="bearish",
                    target_price=target,
                    stop_loss=top * 1.01,
                    confidence=min(1.0, 0.5 + (1.0 - pct_diff / self.tolerance_pct) * 0.5),
                ))

        return matches

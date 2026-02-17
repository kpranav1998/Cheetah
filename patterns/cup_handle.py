from __future__ import annotations

import numpy as np
import pandas as pd

from patterns.base import BasePatternDetector, PatternMatch


class CupHandleDetector(BasePatternDetector):
    """Detects cup and handle pattern (bullish continuation)."""
    name = "cup_handle"

    def __init__(self, min_cup_bars: int = 20, max_cup_bars: int = 120, max_depth_pct: float = 35.0):
        self.min_cup_bars = min_cup_bars
        self.max_cup_bars = max_cup_bars
        self.max_depth_pct = max_depth_pct

    def detect(self, df: pd.DataFrame) -> list[PatternMatch]:
        matches = []

        for cup_len in range(self.min_cup_bars, min(self.max_cup_bars, len(df) - 10), 10):
            for start in range(0, len(df) - cup_len - 10, 5):
                end = start + cup_len
                if end >= len(df):
                    break

                cup = df.iloc[start:end + 1]
                left_rim = cup.iloc[0]["close"]
                right_rim = cup.iloc[-1]["close"]

                # Rims should be at similar levels
                rim_diff = abs(left_rim - right_rim) / left_rim * 100
                if rim_diff > 5.0:
                    continue

                # Cup bottom
                bottom_idx = cup["low"].idxmin()
                bottom_price = cup["low"].min()
                rim_avg = (left_rim + right_rim) / 2
                depth_pct = (rim_avg - bottom_price) / rim_avg * 100

                if depth_pct < 5 or depth_pct > self.max_depth_pct:
                    continue

                # Bottom should be roughly in the middle
                bottom_pos = df.index.get_loc(bottom_idx) - start
                if bottom_pos < cup_len * 0.3 or bottom_pos > cup_len * 0.7:
                    continue

                # Look for handle (small pullback after right rim)
                handle_start = end + 1
                handle_end = min(handle_start + 15, len(df) - 1)
                if handle_start >= len(df):
                    continue

                handle = df.iloc[handle_start:handle_end + 1]
                if handle.empty:
                    continue

                handle_low = handle["low"].min()
                handle_pullback = (right_rim - handle_low) / right_rim * 100

                # Handle pullback should be less than 1/3 of cup depth
                if handle_pullback > depth_pct / 3:
                    continue

                # Breakout above right rim
                breakout_start = handle_end + 1
                if breakout_start >= len(df):
                    continue

                remaining = df.iloc[breakout_start:min(breakout_start + 10, len(df))]
                breakout = remaining[remaining["close"] > rim_avg]
                if breakout.empty:
                    continue

                confirm_ts = breakout.index[0]
                target = rim_avg + (rim_avg - bottom_price)

                matches.append(PatternMatch(
                    pattern_type="cup_handle",
                    start_idx=start,
                    end_idx=df.index.get_loc(confirm_ts),
                    confirmation_timestamp=confirm_ts,
                    direction="bullish",
                    target_price=target,
                    stop_loss=handle_low * 0.99,
                    confidence=min(1.0, 0.4 + (1.0 - rim_diff / 5.0) * 0.3 + (1.0 - abs(bottom_pos / cup_len - 0.5) * 2) * 0.3),
                    metadata={"depth_pct": depth_pct, "handle_pullback_pct": handle_pullback},
                ))

        return matches

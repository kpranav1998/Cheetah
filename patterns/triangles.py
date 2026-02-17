from __future__ import annotations

import numpy as np
import pandas as pd

from patterns.base import BasePatternDetector, PatternMatch
from patterns.support_resistance import find_pivots


class TriangleDetector(BasePatternDetector):
    """Detects ascending, descending, and symmetric triangle patterns."""
    name = "triangle"

    def __init__(self, min_bars: int = 15, max_bars: int = 60, min_touches: int = 2):
        self.min_bars = min_bars
        self.max_bars = max_bars
        self.min_touches = min_touches

    def detect(self, df: pd.DataFrame) -> list[PatternMatch]:
        matches = []
        pivot_highs, pivot_lows = find_pivots(df, left=3, right=3)

        for start in range(0, len(df) - self.min_bars, 5):
            for length in range(self.min_bars, min(self.max_bars, len(df) - start), 5):
                end = start + length
                window = df.iloc[start:end + 1]

                highs_in_window = pivot_highs.iloc[start:end + 1].dropna()
                lows_in_window = pivot_lows.iloc[start:end + 1].dropna()

                if len(highs_in_window) < self.min_touches or len(lows_in_window) < self.min_touches:
                    continue

                # Fit trend lines to highs and lows
                high_positions = [df.index.get_loc(idx) - start for idx in highs_in_window.index]
                low_positions = [df.index.get_loc(idx) - start for idx in lows_in_window.index]

                high_slope, high_intercept = np.polyfit(high_positions, highs_in_window.values, 1)
                low_slope, low_intercept = np.polyfit(low_positions, lows_in_window.values, 1)

                # Classify triangle type
                pattern_type = None
                direction = None

                if abs(high_slope) < 0.01 * high_intercept / length and low_slope > 0:
                    # Flat top, rising bottom = ascending triangle (bullish)
                    pattern_type = "ascending_triangle"
                    direction = "bullish"
                elif high_slope < 0 and abs(low_slope) < 0.01 * low_intercept / length:
                    # Falling top, flat bottom = descending triangle (bearish)
                    pattern_type = "descending_triangle"
                    direction = "bearish"
                elif high_slope < 0 and low_slope > 0:
                    # Converging = symmetric triangle (direction depends on breakout)
                    pattern_type = "symmetric_triangle"
                    direction = "bullish"  # Default; adjusted on breakout

                if pattern_type is None:
                    continue

                # Check for breakout
                if end + 1 >= len(df):
                    continue

                remaining = df.iloc[end + 1:min(end + 10, len(df))]
                apex_high = high_intercept + high_slope * length
                apex_low = low_intercept + low_slope * length

                breakout_up = remaining[remaining["close"] > apex_high]
                breakout_down = remaining[remaining["close"] < apex_low]

                height = high_intercept - low_intercept

                if not breakout_up.empty and direction in ("bullish", "bullish"):
                    confirm_ts = breakout_up.index[0]
                    matches.append(PatternMatch(
                        pattern_type=pattern_type,
                        start_idx=start,
                        end_idx=df.index.get_loc(confirm_ts),
                        confirmation_timestamp=confirm_ts,
                        direction="bullish",
                        target_price=apex_high + height,
                        stop_loss=apex_low * 0.99,
                        confidence=0.6,
                        metadata={"high_slope": high_slope, "low_slope": low_slope},
                    ))
                elif not breakout_down.empty and direction in ("bearish", "bullish"):
                    confirm_ts = breakout_down.index[0]
                    matches.append(PatternMatch(
                        pattern_type=pattern_type,
                        start_idx=start,
                        end_idx=df.index.get_loc(confirm_ts),
                        confirmation_timestamp=confirm_ts,
                        direction="bearish",
                        target_price=apex_low - height,
                        stop_loss=apex_high * 1.01,
                        confidence=0.6,
                        metadata={"high_slope": high_slope, "low_slope": low_slope},
                    ))

        return matches

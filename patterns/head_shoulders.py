from __future__ import annotations

import numpy as np
import pandas as pd

from patterns.base import BasePatternDetector, PatternMatch
from patterns.support_resistance import find_pivots


class HeadShouldersDetector(BasePatternDetector):
    """Detects head & shoulders (bearish) and inverse head & shoulders (bullish)."""
    name = "head_shoulders"

    def __init__(self, tolerance_pct: float = 3.0):
        self.tolerance_pct = tolerance_pct

    def detect(self, df: pd.DataFrame) -> list[PatternMatch]:
        matches = []
        matches.extend(self._detect_hs(df))
        matches.extend(self._detect_inverse_hs(df))
        return matches

    def _detect_hs(self, df: pd.DataFrame) -> list[PatternMatch]:
        """Regular head & shoulders (bearish reversal)."""
        pivot_highs, pivot_lows = find_pivots(df, left=5, right=5)
        high_indices = pivot_highs.dropna().index.tolist()
        matches = []

        for i in range(len(high_indices) - 2):
            ls_idx = df.index.get_loc(high_indices[i])
            h_idx = df.index.get_loc(high_indices[i + 1])
            rs_idx = df.index.get_loc(high_indices[i + 2])

            ls_price = pivot_highs[high_indices[i]]
            h_price = pivot_highs[high_indices[i + 1]]
            rs_price = pivot_highs[high_indices[i + 2]]

            # Head must be higher than both shoulders
            if h_price <= ls_price or h_price <= rs_price:
                continue

            # Shoulders should be at roughly equal heights
            shoulder_diff = abs(ls_price - rs_price) / ls_price * 100
            if shoulder_diff > self.tolerance_pct:
                continue

            # Neckline: connect the lows between shoulders
            between_left = df.iloc[ls_idx:h_idx + 1]
            between_right = df.iloc[h_idx:rs_idx + 1]
            neckline = min(between_left["low"].min(), between_right["low"].min())

            # Confirm: breakout below neckline
            confirm_start = rs_idx + 1
            if confirm_start >= len(df):
                continue

            remaining = df.iloc[confirm_start:min(confirm_start + 20, len(df))]
            breakout = remaining[remaining["close"] < neckline]
            if breakout.empty:
                continue

            confirm_ts = breakout.index[0]
            target = neckline - (h_price - neckline)

            matches.append(PatternMatch(
                pattern_type="head_shoulders",
                start_idx=ls_idx,
                end_idx=df.index.get_loc(confirm_ts),
                confirmation_timestamp=confirm_ts,
                direction="bearish",
                target_price=target,
                stop_loss=h_price * 1.01,
                confidence=min(1.0, 0.6 + (1.0 - shoulder_diff / self.tolerance_pct) * 0.4),
                metadata={"head": h_price, "left_shoulder": ls_price, "right_shoulder": rs_price, "neckline": neckline},
            ))

        return matches

    def _detect_inverse_hs(self, df: pd.DataFrame) -> list[PatternMatch]:
        """Inverse head & shoulders (bullish reversal)."""
        _, pivot_lows = find_pivots(df, left=5, right=5)
        low_indices = pivot_lows.dropna().index.tolist()
        matches = []

        for i in range(len(low_indices) - 2):
            ls_idx = df.index.get_loc(low_indices[i])
            h_idx = df.index.get_loc(low_indices[i + 1])
            rs_idx = df.index.get_loc(low_indices[i + 2])

            ls_price = pivot_lows[low_indices[i]]
            h_price = pivot_lows[low_indices[i + 1]]
            rs_price = pivot_lows[low_indices[i + 2]]

            # Head must be lower than both shoulders
            if h_price >= ls_price or h_price >= rs_price:
                continue

            shoulder_diff = abs(ls_price - rs_price) / ls_price * 100
            if shoulder_diff > self.tolerance_pct:
                continue

            between_left = df.iloc[ls_idx:h_idx + 1]
            between_right = df.iloc[h_idx:rs_idx + 1]
            neckline = max(between_left["high"].max(), between_right["high"].max())

            confirm_start = rs_idx + 1
            if confirm_start >= len(df):
                continue

            remaining = df.iloc[confirm_start:min(confirm_start + 20, len(df))]
            breakout = remaining[remaining["close"] > neckline]
            if breakout.empty:
                continue

            confirm_ts = breakout.index[0]
            target = neckline + (neckline - h_price)

            matches.append(PatternMatch(
                pattern_type="inverse_head_shoulders",
                start_idx=ls_idx,
                end_idx=df.index.get_loc(confirm_ts),
                confirmation_timestamp=confirm_ts,
                direction="bullish",
                target_price=target,
                stop_loss=h_price * 0.99,
                confidence=min(1.0, 0.6 + (1.0 - shoulder_diff / self.tolerance_pct) * 0.4),
                metadata={"head": h_price, "left_shoulder": ls_price, "right_shoulder": rs_price, "neckline": neckline},
            ))

        return matches

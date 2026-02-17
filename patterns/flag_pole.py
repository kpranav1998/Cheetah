from __future__ import annotations

import numpy as np
import pandas as pd

from patterns.base import BasePatternDetector, PatternMatch


class FlagPoleDetector(BasePatternDetector):
    """Detects bull flag and bear flag patterns."""
    name = "flag_pole"

    def __init__(
        self,
        pole_min_bars: int = 5,
        pole_max_bars: int = 25,
        flag_min_bars: int = 5,
        flag_max_bars: int = 20,
        pole_min_move_pct: float = 5.0,
    ):
        self.pole_min_bars = pole_min_bars
        self.pole_max_bars = pole_max_bars
        self.flag_min_bars = flag_min_bars
        self.flag_max_bars = flag_max_bars
        self.pole_min_move_pct = pole_min_move_pct

    def detect(self, df: pd.DataFrame) -> list[PatternMatch]:
        matches = []
        matches.extend(self._detect_bull_flag(df))
        matches.extend(self._detect_bear_flag(df))
        return matches

    def _detect_bull_flag(self, df: pd.DataFrame) -> list[PatternMatch]:
        matches = []
        i = 0

        while i < len(df) - self.pole_min_bars - self.flag_min_bars:
            # Find a strong upward pole
            for pole_len in range(self.pole_min_bars, min(self.pole_max_bars + 1, len(df) - i)):
                pole_start = i
                pole_end = i + pole_len
                if pole_end >= len(df):
                    break

                pole_move = (df.iloc[pole_end]["close"] - df.iloc[pole_start]["close"]) / df.iloc[pole_start]["close"] * 100
                if pole_move < self.pole_min_move_pct:
                    continue

                # Look for consolidation (flag) after the pole
                for flag_len in range(self.flag_min_bars, min(self.flag_max_bars + 1, len(df) - pole_end)):
                    flag_start = pole_end
                    flag_end = pole_end + flag_len
                    if flag_end >= len(df):
                        break

                    flag_data = df.iloc[flag_start:flag_end + 1]
                    flag_range = (flag_data["high"].max() - flag_data["low"].min()) / flag_data["close"].mean() * 100

                    # Flag should retrace less than 50% of pole and be narrower
                    flag_retrace = (df.iloc[pole_end]["close"] - flag_data["low"].min()) / (df.iloc[pole_end]["close"] - df.iloc[pole_start]["close"]) * 100
                    if flag_retrace > 50 or flag_range > pole_move * 0.5:
                        continue

                    # Check for breakout above flag high
                    if flag_end + 1 >= len(df):
                        continue

                    flag_high = flag_data["high"].max()
                    remaining = df.iloc[flag_end + 1:min(flag_end + 10, len(df))]
                    breakout = remaining[remaining["close"] > flag_high]

                    if breakout.empty:
                        continue

                    confirm_ts = breakout.index[0]
                    # Target: pole height projected from breakout
                    pole_height = df.iloc[pole_end]["close"] - df.iloc[pole_start]["close"]
                    target = flag_high + pole_height

                    matches.append(PatternMatch(
                        pattern_type="bull_flag",
                        start_idx=pole_start,
                        end_idx=df.index.get_loc(confirm_ts),
                        confirmation_timestamp=confirm_ts,
                        direction="bullish",
                        target_price=target,
                        stop_loss=flag_data["low"].min() * 0.99,
                        confidence=min(1.0, 0.5 + pole_move / 20),
                        metadata={"pole_move_pct": pole_move, "flag_retrace_pct": flag_retrace},
                    ))
                    i = flag_end  # Skip ahead
                    break
                else:
                    continue
                break
            i += 1

        return matches

    def _detect_bear_flag(self, df: pd.DataFrame) -> list[PatternMatch]:
        matches = []
        i = 0

        while i < len(df) - self.pole_min_bars - self.flag_min_bars:
            for pole_len in range(self.pole_min_bars, min(self.pole_max_bars + 1, len(df) - i)):
                pole_start = i
                pole_end = i + pole_len
                if pole_end >= len(df):
                    break

                pole_move = (df.iloc[pole_start]["close"] - df.iloc[pole_end]["close"]) / df.iloc[pole_start]["close"] * 100
                if pole_move < self.pole_min_move_pct:
                    continue

                for flag_len in range(self.flag_min_bars, min(self.flag_max_bars + 1, len(df) - pole_end)):
                    flag_start = pole_end
                    flag_end = pole_end + flag_len
                    if flag_end >= len(df):
                        break

                    flag_data = df.iloc[flag_start:flag_end + 1]
                    flag_range = (flag_data["high"].max() - flag_data["low"].min()) / flag_data["close"].mean() * 100

                    flag_retrace = (flag_data["high"].max() - df.iloc[pole_end]["close"]) / (df.iloc[pole_start]["close"] - df.iloc[pole_end]["close"]) * 100
                    if flag_retrace > 50 or flag_range > pole_move * 0.5:
                        continue

                    if flag_end + 1 >= len(df):
                        continue

                    flag_low = flag_data["low"].min()
                    remaining = df.iloc[flag_end + 1:min(flag_end + 10, len(df))]
                    breakout = remaining[remaining["close"] < flag_low]

                    if breakout.empty:
                        continue

                    confirm_ts = breakout.index[0]
                    pole_height = df.iloc[pole_start]["close"] - df.iloc[pole_end]["close"]
                    target = flag_low - pole_height

                    matches.append(PatternMatch(
                        pattern_type="bear_flag",
                        start_idx=pole_start,
                        end_idx=df.index.get_loc(confirm_ts),
                        confirmation_timestamp=confirm_ts,
                        direction="bearish",
                        target_price=target,
                        stop_loss=flag_data["high"].max() * 1.01,
                        confidence=min(1.0, 0.5 + pole_move / 20),
                        metadata={"pole_move_pct": pole_move, "flag_retrace_pct": flag_retrace},
                    ))
                    i = flag_end
                    break
                else:
                    continue
                break
            i += 1

        return matches

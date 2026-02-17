from enum import Enum


class Timeframe(Enum):
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    DAILY = "1d"
    WEEKLY = "1wk"

    @property
    def yfinance_interval(self) -> str:
        return self.value

    @property
    def minutes(self) -> int:
        mapping = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "1d": 375, "1wk": 1875,
        }
        return mapping[self.value]

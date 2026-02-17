from __future__ import annotations

from datetime import date, timedelta

# NSE trading holidays for 2024-2026 (update as needed)
NSE_HOLIDAYS: set[date] = {
    # 2024
    date(2024, 1, 26), date(2024, 3, 8), date(2024, 3, 25), date(2024, 3, 29),
    date(2024, 4, 11), date(2024, 4, 14), date(2024, 4, 17), date(2024, 4, 21),
    date(2024, 5, 1), date(2024, 5, 23), date(2024, 6, 17), date(2024, 7, 17),
    date(2024, 8, 15), date(2024, 10, 2), date(2024, 10, 12), date(2024, 11, 1),
    date(2024, 11, 15), date(2024, 12, 25),
    # 2025
    date(2025, 1, 26), date(2025, 2, 26), date(2025, 3, 14), date(2025, 3, 31),
    date(2025, 4, 10), date(2025, 4, 14), date(2025, 4, 18), date(2025, 5, 1),
    date(2025, 8, 15), date(2025, 8, 27), date(2025, 10, 2), date(2025, 10, 21),
    date(2025, 10, 22), date(2025, 11, 5), date(2025, 12, 25),
    # 2026 (tentative)
    date(2026, 1, 26), date(2026, 3, 10), date(2026, 3, 19), date(2026, 4, 3),
    date(2026, 4, 14), date(2026, 5, 1), date(2026, 8, 15), date(2026, 10, 2),
    date(2026, 12, 25),
}


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in NSE_HOLIDAYS


def next_trading_day(d: date) -> date:
    d = d + timedelta(days=1)
    while not is_trading_day(d):
        d = d + timedelta(days=1)
    return d


def get_monthly_expiry(year: int, month: int) -> date:
    """Last Thursday of the month (or Wednesday if Thursday is a holiday)."""
    # Start from last day and walk backwards to find last Thursday
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)

    d = last_day
    while d.weekday() != 3:  # Thursday = 3
        d -= timedelta(days=1)

    # If Thursday is a holiday, expiry moves to Wednesday
    if d in NSE_HOLIDAYS:
        d -= timedelta(days=1)
    return d


def get_weekly_expiry(d: date) -> date:
    """Next Thursday on or after date d (for NIFTY/BANKNIFTY weekly options)."""
    days_ahead = (3 - d.weekday()) % 7
    if days_ahead == 0 and d.weekday() == 3:
        expiry = d
    else:
        expiry = d + timedelta(days=days_ahead)

    if expiry in NSE_HOLIDAYS:
        expiry -= timedelta(days=1)
    return expiry

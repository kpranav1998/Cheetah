from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OptionLegPayoff:
    option_type: str  # "CE" or "PE"
    strike: float
    premium: float
    action: str  # "BUY" or "SELL"
    lots: int = 1
    lot_size: int = 25


def calculate_payoff(
    legs: list[OptionLegPayoff],
    spot_range: np.ndarray | None = None,
    center_price: float | None = None,
) -> pd.DataFrame:
    """Calculate combined payoff for multi-leg options strategy."""
    if spot_range is None:
        strikes = [leg.strike for leg in legs]
        center = center_price or np.mean(strikes)
        spread = max(strikes) - min(strikes) or center * 0.1
        spot_range = np.linspace(center - spread * 2, center + spread * 2, 200)

    payoff = np.zeros_like(spot_range, dtype=float)

    for leg in legs:
        qty = leg.lots * leg.lot_size
        multiplier = 1 if leg.action == "BUY" else -1

        if leg.option_type == "CE":
            intrinsic = np.maximum(spot_range - leg.strike, 0)
        else:
            intrinsic = np.maximum(leg.strike - spot_range, 0)

        leg_payoff = multiplier * (intrinsic - leg.premium) * qty
        payoff += leg_payoff

    result = pd.DataFrame({"spot": spot_range, "payoff": payoff})

    # Key metrics
    result.attrs["max_profit"] = float(payoff.max())
    result.attrs["max_loss"] = float(payoff.min())

    # Breakeven points
    sign_changes = np.where(np.diff(np.sign(payoff)))[0]
    breakevens = []
    for idx in sign_changes:
        x1, x2 = spot_range[idx], spot_range[idx + 1]
        y1, y2 = payoff[idx], payoff[idx + 1]
        if y2 != y1:
            be = x1 - y1 * (x2 - x1) / (y2 - y1)
            breakevens.append(float(be))
    result.attrs["breakevens"] = breakevens

    return result

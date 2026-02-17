from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from strategies.base import Signal
from strategies.options.base_options import BaseOptionsStrategy, OptionsLeg, OptionsSignal
from strategies.registry import register


@register
class StraddleStrategy(BaseOptionsStrategy):
    """Buy/sell ATM call + ATM put. Profits from large moves (long) or range-bound (short)."""
    name = "straddle"

    def __init__(self):
        self.direction = "long"  # "long" or "short"
        self.profit_target_pct = 50.0
        self.stop_loss_pct = 30.0

    def configure(self, params: dict[str, Any]) -> None:
        self.direction = params.get("direction", self.direction)
        self.profit_target_pct = params.get("profit_target_pct", self.profit_target_pct)
        self.stop_loss_pct = params.get("stop_loss_pct", self.stop_loss_pct)

    def select_strikes(self, underlying_price, options_chain, params) -> list[OptionsLeg]:
        strikes = options_chain["strike"].values
        atm_strike = float(strikes[abs(strikes - underlying_price).argmin()])
        expiry = options_chain.attrs.get("expiry", date.today())
        action = "BUY" if self.direction == "long" else "SELL"

        return [
            OptionsLeg(option_type="CE", strike=atm_strike, expiry=expiry, action=action),
            OptionsLeg(option_type="PE", strike=atm_strike, expiry=expiry, action=action),
        ]

    def entry_condition(self, df, chain) -> bool:
        return True

    def exit_condition(self, df, chain, entry_signal, current_pnl) -> bool:
        if entry_signal.max_profit and current_pnl >= entry_signal.max_profit * (self.profit_target_pct / 100):
            return True
        if entry_signal.max_loss and current_pnl <= entry_signal.max_loss * (self.stop_loss_pct / 100):
            return True
        return False

    def generate_signals(self, df) -> list[Signal]:
        return []

    def parameter_space(self) -> dict[str, list[Any]]:
        return {"direction": ["long", "short"]}

    def get_params(self) -> dict[str, Any]:
        return {"direction": self.direction, "profit_target_pct": self.profit_target_pct}

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from strategies.base import Signal
from strategies.options.base_options import BaseOptionsStrategy, OptionsLeg, OptionsSignal
from strategies.registry import register


@register
class StrangleStrategy(BaseOptionsStrategy):
    """Buy/sell OTM call + OTM put. Cheaper than straddle, needs bigger move."""
    name = "strangle"

    def __init__(self):
        self.direction = "short"
        self.otm_distance_pct = 3.0
        self.profit_target_pct = 50.0
        self.stop_loss_pct = 100.0

    def configure(self, params: dict[str, Any]) -> None:
        self.direction = params.get("direction", self.direction)
        self.otm_distance_pct = params.get("otm_distance_pct", self.otm_distance_pct)
        self.profit_target_pct = params.get("profit_target_pct", self.profit_target_pct)
        self.stop_loss_pct = params.get("stop_loss_pct", self.stop_loss_pct)

    def select_strikes(self, underlying_price, options_chain, params) -> list[OptionsLeg]:
        strikes = options_chain["strike"].values
        call_strike = float(strikes[abs(strikes - underlying_price * (1 + self.otm_distance_pct / 100)).argmin()])
        put_strike = float(strikes[abs(strikes - underlying_price * (1 - self.otm_distance_pct / 100)).argmin()])
        expiry = options_chain.attrs.get("expiry", date.today())
        action = "BUY" if self.direction == "long" else "SELL"

        return [
            OptionsLeg(option_type="CE", strike=call_strike, expiry=expiry, action=action),
            OptionsLeg(option_type="PE", strike=put_strike, expiry=expiry, action=action),
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
        return {"otm_distance_pct": [2.0, 3.0, 5.0], "direction": ["long", "short"]}

    def get_params(self) -> dict[str, Any]:
        return {"direction": self.direction, "otm_distance_pct": self.otm_distance_pct}

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from strategies.base import Signal
from strategies.options.base_options import BaseOptionsStrategy, OptionsLeg, OptionsSignal
from strategies.registry import register


@register
class BearPutSpreadStrategy(BaseOptionsStrategy):
    """Buy ATM put, sell OTM put. Directional bearish with limited risk."""
    name = "bear_put_spread"

    def __init__(self):
        self.spread_width_pct = 3.0
        self.profit_target_pct = 70.0
        self.stop_loss_pct = 50.0

    def configure(self, params: dict[str, Any]) -> None:
        self.spread_width_pct = params.get("spread_width_pct", self.spread_width_pct)
        self.profit_target_pct = params.get("profit_target_pct", self.profit_target_pct)
        self.stop_loss_pct = params.get("stop_loss_pct", self.stop_loss_pct)

    def select_strikes(self, underlying_price, options_chain, params) -> list[OptionsLeg]:
        strikes = options_chain["strike"].values
        long_strike = float(strikes[abs(strikes - underlying_price).argmin()])
        short_strike = float(strikes[abs(strikes - underlying_price * (1 - self.spread_width_pct / 100)).argmin()])
        expiry = options_chain.attrs.get("expiry", date.today())

        return [
            OptionsLeg(option_type="PE", strike=long_strike, expiry=expiry, action="BUY"),
            OptionsLeg(option_type="PE", strike=short_strike, expiry=expiry, action="SELL"),
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
        return {"spread_width_pct": [2.0, 3.0, 5.0]}

    def get_params(self) -> dict[str, Any]:
        return {"spread_width_pct": self.spread_width_pct, "profit_target_pct": self.profit_target_pct}

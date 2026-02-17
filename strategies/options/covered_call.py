from datetime import date
from typing import Any

from strategies.base import Signal
from strategies.options.base_options import BaseOptionsStrategy, OptionsLeg
from strategies.registry import register


@register
class CoveredCallStrategy(BaseOptionsStrategy):
    """Hold stock + sell OTM call. Generates income, caps upside."""
    name = "covered_call"

    def __init__(self):
        self.otm_distance_pct = 3.0
        self.profit_target_pct = 80.0
        self.stop_loss_pct = 50.0

    def configure(self, params):
        self.otm_distance_pct = params.get("otm_distance_pct", self.otm_distance_pct)

    def select_strikes(self, underlying_price, options_chain, params):
        strikes = options_chain["strike"].values
        call_strike = float(strikes[abs(strikes - underlying_price * (1 + self.otm_distance_pct / 100)).argmin()])
        expiry = options_chain.attrs.get("expiry", date.today())

        return [
            OptionsLeg(option_type="CE", strike=call_strike, expiry=expiry, action="SELL"),
        ]

    def entry_condition(self, df, chain):
        return True

    def exit_condition(self, df, chain, entry_signal, current_pnl):
        if entry_signal.max_profit and current_pnl >= entry_signal.max_profit * (self.profit_target_pct / 100):
            return True
        return False

    def generate_signals(self, df):
        return []

    def parameter_space(self):
        return {"otm_distance_pct": [2.0, 3.0, 5.0]}

    def get_params(self):
        return {"otm_distance_pct": self.otm_distance_pct}

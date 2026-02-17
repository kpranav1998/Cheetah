from datetime import date
from typing import Any

from strategies.base import Signal
from strategies.options.base_options import BaseOptionsStrategy, OptionsLeg
from strategies.registry import register


@register
class ProtectivePutStrategy(BaseOptionsStrategy):
    """Hold stock + buy OTM put. Insurance against downside."""
    name = "protective_put"

    def __init__(self):
        self.otm_distance_pct = 5.0
        self.profit_target_pct = 50.0
        self.stop_loss_pct = 100.0

    def configure(self, params):
        self.otm_distance_pct = params.get("otm_distance_pct", self.otm_distance_pct)

    def select_strikes(self, underlying_price, options_chain, params):
        strikes = options_chain["strike"].values
        put_strike = float(strikes[abs(strikes - underlying_price * (1 - self.otm_distance_pct / 100)).argmin()])
        expiry = options_chain.attrs.get("expiry", date.today())

        return [
            OptionsLeg(option_type="PE", strike=put_strike, expiry=expiry, action="BUY"),
        ]

    def entry_condition(self, df, chain):
        return True

    def exit_condition(self, df, chain, entry_signal, current_pnl):
        return False  # Typically hold until expiry

    def generate_signals(self, df):
        return []

    def parameter_space(self):
        return {"otm_distance_pct": [3.0, 5.0, 7.0]}

    def get_params(self):
        return {"otm_distance_pct": self.otm_distance_pct}

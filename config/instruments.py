from __future__ import annotations

NSE_LOT_SIZES: dict[str, int] = {
    "NIFTY": 25,
    "BANKNIFTY": 15,
    "FINNIFTY": 25,
    "RELIANCE": 250,
    "TCS": 150,
    "INFY": 300,
    "HDFCBANK": 550,
    "ICICIBANK": 1375,
    "SBIN": 1500,
    "TATAMOTORS": 1400,
    "BHARTIARTL": 950,
    "ITC": 1600,
    "KOTAKBANK": 400,
    "LT": 300,
    "AXISBANK": 600,
    "MARUTI": 100,
    "BAJFINANCE": 125,
    "TATASTEEL": 5500,
    "HINDALCO": 1400,
    "WIPRO": 1500,
}

# Zerodha-accurate charges for Indian markets
CHARGES: dict[str, dict[str, float]] = {
    "equity_intraday": {
        "brokerage_per_order": 20.0,
        "stt_sell_pct": 0.025,
        "exchange_txn_pct": 0.00345,
        "gst_on_brokerage_pct": 18.0,
        "sebi_per_crore": 10.0,
        "stamp_buy_pct": 0.003,
    },
    "equity_delivery": {
        "brokerage_per_order": 0.0,
        "stt_both_pct": 0.1,
        "exchange_txn_pct": 0.00345,
        "gst_on_brokerage_pct": 18.0,
        "sebi_per_crore": 10.0,
        "stamp_buy_pct": 0.015,
    },
    "futures": {
        "brokerage_per_order": 20.0,
        "stt_sell_pct": 0.02,
        "exchange_txn_pct": 0.002,
        "gst_on_brokerage_pct": 18.0,
        "sebi_per_crore": 10.0,
        "stamp_buy_pct": 0.002,
    },
    "options_buy": {
        "brokerage_per_order": 20.0,
        "stt_sell_pct": 0.1,
        "exchange_txn_pct": 0.053,
        "gst_on_brokerage_pct": 18.0,
        "sebi_per_crore": 10.0,
        "stamp_buy_pct": 0.003,
    },
    "options_sell": {
        "brokerage_per_order": 20.0,
        "stt_sell_pct": 0.1,
        "exchange_txn_pct": 0.053,
        "gst_on_brokerage_pct": 18.0,
        "sebi_per_crore": 10.0,
        "stamp_buy_pct": 0.003,
    },
}


def get_lot_size(symbol: str) -> int:
    return NSE_LOT_SIZES.get(symbol.upper(), 1)

import pandas as pd
from kiteconnect import KiteConnect

from config.settings import settings


class OptionsChainFetcher:
    """Fetch live options chain from Kite Connect."""

    def __init__(self):
        self.kite = KiteConnect(api_key=settings.kite_api_key)
        if settings.kite_access_token:
            self.kite.set_access_token(settings.kite_access_token)

    def fetch_chain(
        self,
        underlying: str,
        exchange: str = "NFO",
    ) -> pd.DataFrame:
        """
        Fetch full options chain for an underlying from instrument list.
        Returns DataFrame with strike, option_type, tradingsymbol, instrument_token, etc.
        """
        instruments = self.kite.instruments(exchange)
        df = pd.DataFrame(instruments)

        # Filter for the underlying
        options = df[
            (df["name"] == underlying)
            & (df["instrument_type"].isin(["CE", "PE"]))
        ].copy()

        if options.empty:
            return options

        options = options.sort_values(["expiry", "strike", "instrument_type"])
        return options[
            ["tradingsymbol", "instrument_token", "strike", "instrument_type",
             "expiry", "lot_size", "tick_size"]
        ].reset_index(drop=True)

    def fetch_chain_with_prices(
        self,
        underlying: str,
        expiry=None,
        exchange: str = "NFO",
    ) -> pd.DataFrame:
        """Fetch chain with live prices (LTP, OI, volume)."""
        chain = self.fetch_chain(underlying, exchange)

        if expiry is not None:
            chain = chain[chain["expiry"] == expiry]

        if chain.empty:
            return chain

        # Get quotes for all options in the chain
        symbols = [f"{exchange}:{ts}" for ts in chain["tradingsymbol"]]

        # Kite API allows max ~500 symbols per call
        all_quotes = {}
        for i in range(0, len(symbols), 500):
            batch = symbols[i:i + 500]
            quotes = self.kite.quote(batch)
            all_quotes.update(quotes)

        # Merge quote data into chain
        ltps = []
        ois = []
        volumes = []
        for _, row in chain.iterrows():
            key = f"{exchange}:{row['tradingsymbol']}"
            quote = all_quotes.get(key, {})
            ltps.append(quote.get("last_price", 0))
            ois.append(quote.get("oi", 0))
            volumes.append(quote.get("volume", 0))

        chain["ltp"] = ltps
        chain["oi"] = ois
        chain["volume"] = volumes

        return chain

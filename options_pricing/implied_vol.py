import numpy as np

from options_pricing.black_scholes import call_price, put_price, vega as bs_vega


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "CE",
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> float:
    """Calculate implied volatility using Newton-Raphson method."""
    sigma = 0.3  # Initial guess

    price_func = call_price if option_type == "CE" else put_price

    for _ in range(max_iterations):
        price = price_func(S, K, T, r, sigma)
        diff = price - market_price

        if abs(diff) < tolerance:
            return sigma

        v = bs_vega(S, K, T, r, sigma) * 100  # Convert back from per-1% to per-1
        if abs(v) < 1e-10:
            break

        sigma -= diff / v
        sigma = max(sigma, 0.01)  # Floor at 1%
        sigma = min(sigma, 5.0)   # Cap at 500%

    return sigma

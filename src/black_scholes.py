"""Black–Scholes prices for European options (no dividends)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def d1(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    """Black–Scholes d1."""
    return (np.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))


def d2(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    """Black–Scholes d2 = d1 - sigma sqrt(T)."""
    return d1(s0, k, r, sigma, t) - sigma * np.sqrt(t)


def call_price(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    """European call: S0 N(d1) - K e^{-rT} N(d2)."""
    return s0 * norm.cdf(d1(s0, k, r, sigma, t)) - k * np.exp(-r * t) * norm.cdf(
        d2(s0, k, r, sigma, t)
    )


def put_price(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    """European put: K e^{-rT} N(-d2) - S0 N(-d1)."""
    return k * np.exp(-r * t) * norm.cdf(-d2(s0, k, r, sigma, t)) - s0 * norm.cdf(
        -d1(s0, k, r, sigma, t)
    )

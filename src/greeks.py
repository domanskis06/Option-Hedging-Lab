"""Black–Scholes Greeks for European options (no dividends)."""

from __future__ import annotations

import math

from scipy.stats import norm

from src.black_scholes import d1, d2


def call_delta(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    """Call delta = N(d1). Hedge ratio for a short call."""
    return norm.cdf(d1(s0, k, r, sigma, t))


def put_delta(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    """Put delta = N(d1) - 1."""
    return norm.cdf(d1(s0, k, r, sigma, t)) - 1.0


def gamma(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    """Gamma = n(d1) / (S0 sigma sqrt(T)). Same for call and put."""
    return norm.pdf(d1(s0, k, r, sigma, t)) / (s0 * sigma * math.sqrt(t))


def vega(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    """Raw vega = S0 n(d1) sqrt(T). Same for call and put."""
    return s0 * norm.pdf(d1(s0, k, r, sigma, t)) * math.sqrt(t)


def call_theta(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    """Call theta (calendar-time convention)."""
    return (
        -(s0 * norm.pdf(d1(s0, k, r, sigma, t)) * sigma) / (2.0 * math.sqrt(t))
        - r * k * math.exp(-r * t) * norm.cdf(d2(s0, k, r, sigma, t))
    )


def put_theta(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    """Put theta (calendar-time convention)."""
    return (
        -(s0 * norm.pdf(d1(s0, k, r, sigma, t)) * sigma) / (2.0 * math.sqrt(t))
        + r * k * math.exp(-r * t) * norm.cdf(-d2(s0, k, r, sigma, t))
    )


def call_rho(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    """Call rho = K T e^{-rT} N(d2)."""
    return k * t * math.exp(-r * t) * norm.cdf(d2(s0, k, r, sigma, t))


def put_rho(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    """Put rho = -K T e^{-rT} N(-d2)."""
    return -k * t * math.exp(-r * t) * norm.cdf(-d2(s0, k, r, sigma, t))

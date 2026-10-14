"""Cox–Ross–Rubinstein binomial trees for European and American options."""

from __future__ import annotations

import math

import numpy as np

from src.payoffs import call_payoff, put_payoff


def crr_european_price(
    option_type: str,
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_steps: int,
) -> float:
    """
    Price a European call/put on a CRR tree (no dividends).

    Uses u = exp(sigma sqrt(dt)), d = 1/u, and risk-neutral p*, then
    backward induction with discounted continuation only.
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    dt = t / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p_star = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    payoff_fn = call_payoff if option_type == "call" else put_payoff

    vals = np.zeros(n_steps + 1)
    for j in range(n_steps + 1):
        spot = s0 * (u**j) * (d ** (n_steps - j))
        vals[j] = payoff_fn(spot, k)

    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            vals[j] = disc * (p_star * vals[j + 1] + (1.0 - p_star) * vals[j])

    return float(vals[0])


def crr_call(s0: float, k: float, r: float, sigma: float, t: float, n_steps: int) -> float:
    """European call via CRR."""
    return crr_european_price("call", s0, k, r, sigma, t, n_steps)


def crr_put(s0: float, k: float, r: float, sigma: float, t: float, n_steps: int) -> float:
    """European put via CRR."""
    return crr_european_price("put", s0, k, r, sigma, t, n_steps)


def crr_american_price(
    option_type: str,
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_steps: int,
) -> float:
    """
    Price an American call/put on a CRR tree.

    Same tree as European, but at each node:
        V = max(exercise_payoff, continuation_value).
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    dt = t / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p_star = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    payoff_fn = call_payoff if option_type == "call" else put_payoff

    vals = np.zeros(n_steps + 1)
    for j in range(n_steps + 1):
        spot = s0 * (u**j) * (d ** (n_steps - j))
        vals[j] = payoff_fn(spot, k)

    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            continuation = disc * (p_star * vals[j + 1] + (1.0 - p_star) * vals[j])
            spot = s0 * (u**j) * (d ** (i - j))
            exercise = float(payoff_fn(spot, k))
            vals[j] = max(continuation, exercise)

    return float(vals[0])


def crr_american_put(
    s0: float, k: float, r: float, sigma: float, t: float, n_steps: int
) -> float:
    """American put via CRR."""
    return crr_american_price("put", s0, k, r, sigma, t, n_steps)


def crr_american_call(
    s0: float, k: float, r: float, sigma: float, t: float, n_steps: int
) -> float:
    """American call via CRR."""
    return crr_american_price("call", s0, k, r, sigma, t, n_steps)

"""European option terminal payoffs."""

from __future__ import annotations

import numpy as np


def call_payoff(spot: float, strike: float) -> float:
    """Return max(S - K, 0). Accepts scalars or NumPy arrays."""
    return np.maximum(spot - strike, 0.0)


def put_payoff(spot: float, strike: float) -> float:
    """Return max(K - S, 0). Accepts scalars or NumPy arrays."""
    return np.maximum(strike - spot, 0.0)

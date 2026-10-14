"""
Risk metrics on a PnL / P&L sample (historical / Monte Carlo style).

VaR at level alpha (e.g. 0.95): loss quantile such that P(L > VaR) ~= 1-alpha,
where loss L = -PnL (positive loss = negative profit).

Expected Shortfall (ES / CVaR): average loss beyond VaR.
"""

from __future__ import annotations

import numpy as np


def var_abs(pnl: np.ndarray, alpha: float = 0.95) -> float:
    """
    Absolute VaR of a PnL sample: quantile of losses L=-PnL at probability alpha.

    Example: alpha=0.95 → 95% VaR = 95th percentile of losses
    (equivalently -5th percentile of PnL).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    losses = -np.asarray(pnl, dtype=float)
    return float(np.quantile(losses, alpha))


def expected_shortfall(pnl: np.ndarray, alpha: float = 0.95) -> float:
    """
    Expected Shortfall: mean loss conditional on loss >= VaR_alpha.
    """
    losses = -np.asarray(pnl, dtype=float)
    threshold = var_abs(pnl, alpha)
    tail = losses[losses >= threshold]
    if tail.size == 0:
        return float(threshold)
    return float(tail.mean())

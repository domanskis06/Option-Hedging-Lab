"""Discrete delta-hedging simulation for a short European call."""

from __future__ import annotations

import math

import numpy as np

from src.black_scholes import call_price
from src.gbm import simulate_gbm_paths
from src.greeks import call_delta
from src.payoffs import call_payoff


def simulate_delta_hedge(
    s0: float,
    k: float,
    r: float,
    sigma_real: float,
    sigma_hedge: float,
    t: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    trade_cost_bps: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """
    Simulate discrete delta-hedging of a short European call.

    At t=0: sell the call (receive BS premium with ``sigma_hedge``), buy
    ``call_delta`` shares. Rebalance delta on a grid; cash grows at rate ``r``.
    Stock paths use GBM with ``sigma_real``. At maturity, liquidate the hedge
    and pay the call payoff.

    Parameters
    ----------
    sigma_real :
        Volatility used to simulate paths.
    sigma_hedge :
        Volatility used for premium and delta (misspecification if different).
    trade_cost_bps :
        Proportional cost in basis points on traded notional.
        Cost = |shares_traded| * S * (bps * 1e-4).

    Returns
    -------
    dict with keys ``pnl`` (n_paths,), ``mean_pnl``, ``std_pnl``.
    """
    dt = t / n_steps
    paths = simulate_gbm_paths(s0, r, sigma_real, t, n_steps, n_paths, seed)

    premiums = call_price(paths[:, 0], k, r, sigma_hedge, t)
    shares = call_delta(paths[:, 0], k, r, sigma_hedge, t)
    cash = premiums - shares * paths[:, 0]
    cash -= np.abs(shares) * paths[:, 0] * (trade_cost_bps * 1e-4)

    for m in range(1, n_steps):
        cash *= math.exp(r * dt)
        tau = t - m * dt
        new_deltas = call_delta(paths[:, m], k, r, sigma_hedge, tau)
        traded = new_deltas - shares
        cash -= traded * paths[:, m]
        cash -= np.abs(traded) * paths[:, m] * (trade_cost_bps * 1e-4)
        shares = new_deltas

    cash *= math.exp(r * dt)
    payoff = call_payoff(paths[:, n_steps], k)
    cash += shares * paths[:, n_steps]
    cash -= np.abs(shares) * paths[:, n_steps] * (trade_cost_bps * 1e-4)
    cash -= payoff

    pnl = cash
    return {"pnl": pnl, "mean_pnl": float(np.mean(pnl)), "std_pnl": float(np.std(pnl))}

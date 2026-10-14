"""Geometric Brownian Motion path simulation (risk-neutral drift)."""

from __future__ import annotations

import numpy as np


def simulate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate GBM paths under the risk-neutral measure:

        dS = r S dt + sigma S dW

    Exact one-step update:

        S_{t+dt} = S_t * exp((r - 0.5 sigma^2) dt + sigma sqrt(dt) Z),
        Z ~ N(0, 1).

    Returns
    -------
    paths : ndarray, shape (n_paths, n_steps + 1)
        paths[:, 0] == s0; paths[:, -1] are terminal prices S_T.
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = s0
    for i in range(n_steps):
        z = rng.standard_normal(n_paths)
        paths[:, i + 1] = paths[:, i] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )
    return paths

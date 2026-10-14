"""Tests for Step 1. Run: python -m pytest tests/test_gbm.py -q"""

import numpy as np
import pytest

from src.gbm import simulate_gbm_paths


def test_shape_and_start():
    paths = simulate_gbm_paths(
        s0=100.0, r=0.05, sigma=0.2, t=1.0, n_steps=10, n_paths=5, seed=42
    )
    assert paths.shape == (5, 11)
    assert np.allclose(paths[:, 0], 100.0)


def test_positive_prices():
    paths = simulate_gbm_paths(
        s0=100.0, r=0.01, sigma=0.3, t=1.0, n_steps=50, n_paths=200, seed=1
    )
    assert np.all(paths > 0)


def test_mean_close_to_risk_neutral_expectation():
    """
    Under GBM risk-neutral dynamics:
        E[S_T] = S0 * exp(r * T)

    With many paths the sample mean should be close.
    """
    s0, r, t = 100.0, 0.05, 1.0
    paths = simulate_gbm_paths(
        s0=s0, r=r, sigma=0.2, t=t, n_steps=100, n_paths=50_000, seed=7
    )
    expected = s0 * np.exp(r * t)
    sample_mean = paths[:, -1].mean()
    # allow ~2% relative error with this many paths
    assert abs(sample_mean - expected) / expected < 0.02


def test_zero_vol_is_deterministic():
    """If sigma=0, every path is S0 * exp(r*t) at the end (no randomness)."""
    s0, r, t = 100.0, 0.05, 1.0
    paths = simulate_gbm_paths(
        s0=s0, r=r, sigma=0.0, t=t, n_steps=20, n_paths=10, seed=0
    )
    expected = s0 * np.exp(r * t)
    assert np.allclose(paths[:, -1], expected)

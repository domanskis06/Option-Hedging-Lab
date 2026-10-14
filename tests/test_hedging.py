"""Tests for Step 5. Run: python -m pytest tests/test_hedging.py -q"""

import numpy as np
import pytest

from src.hedging import simulate_delta_hedge


def test_return_keys_and_shapes():
    out = simulate_delta_hedge(
        s0=100,
        k=100,
        r=0.05,
        sigma_real=0.2,
        sigma_hedge=0.2,
        t=1.0,
        n_steps=50,
        n_paths=200,
        seed=0,
    )
    assert "pnl" in out and "mean_pnl" in out and "std_pnl" in out
    assert out["pnl"].shape == (200,)
    assert np.isfinite(out["mean_pnl"])
    assert out["std_pnl"] >= 0


def test_frequent_hedge_mean_near_zero_when_vols_match():
    """
    In the BS world, continuous delta-hedging of a fairly priced short call
    has expected PnL ~ 0. With many steps and matched vols, mean should be small.
    """
    out = simulate_delta_hedge(
        s0=100,
        k=100,
        r=0.05,
        sigma_real=0.2,
        sigma_hedge=0.2,
        t=1.0,
        n_steps=252,
        n_paths=3000,
        seed=42,
        trade_cost_bps=0.0,
    )
    # allow some MC noise
    assert abs(out["mean_pnl"]) < 0.35


def test_more_steps_reduces_pnl_volatility():
    common = dict(
        s0=100,
        k=100,
        r=0.05,
        sigma_real=0.2,
        sigma_hedge=0.2,
        t=1.0,
        n_paths=2500,
        seed=7,
        trade_cost_bps=0.0,
    )
    coarse = simulate_delta_hedge(n_steps=12, **common)
    fine = simulate_delta_hedge(n_steps=252, **common)
    assert fine["std_pnl"] < coarse["std_pnl"]


def test_transaction_costs_pull_mean_pnl_down():
    common = dict(
        s0=100,
        k=100,
        r=0.05,
        sigma_real=0.2,
        sigma_hedge=0.2,
        t=1.0,
        n_steps=100,
        n_paths=2000,
        seed=1,
    )
    free = simulate_delta_hedge(trade_cost_bps=0.0, **common)
    costly = simulate_delta_hedge(trade_cost_bps=10.0, **common)  # 10 bps
    assert costly["mean_pnl"] < free["mean_pnl"]


def test_wrong_hedge_vol_worsens_result():
    """
    Hedging with badly wrong sigma should not look like the matched case:
    either mean drifts away and/or dispersion grows. We check |mean| gets worse.
    """
    matched = simulate_delta_hedge(
        s0=100,
        k=100,
        r=0.05,
        sigma_real=0.2,
        sigma_hedge=0.2,
        t=1.0,
        n_steps=100,
        n_paths=2500,
        seed=3,
    )
    mismatched = simulate_delta_hedge(
        s0=100,
        k=100,
        r=0.05,
        sigma_real=0.2,
        sigma_hedge=0.4,  # hedge with double vol
        t=1.0,
        n_steps=100,
        n_paths=2500,
        seed=3,
    )
    assert abs(mismatched["mean_pnl"]) > abs(matched["mean_pnl"])

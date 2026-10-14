"""Tests for Step 2. Run: python -m pytest tests/test_black_scholes.py -q"""

import math

import pytest

from src.black_scholes import call_price, d1, d2, put_price


def test_d2_equals_d1_minus_vol_term():
    s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
    assert d2(s0, k, r, sigma, t) == pytest.approx(
        d1(s0, k, r, sigma, t) - sigma * math.sqrt(t)
    )


def test_atm_call_known_value():
    # Classic textbook-ish ATM parameters
    c = call_price(s0=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0)
    # Reference ~ 10.4506
    assert c == pytest.approx(10.450583572185565, rel=1e-6)


def test_put_call_parity():
    s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
    c = call_price(s0, k, r, sigma, t)
    p = put_price(s0, k, r, sigma, t)
    # C - P = S0 - K*exp(-rT)   (no dividends)
    lhs = c - p
    rhs = s0 - k * math.exp(-r * t)
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_call_decreases_with_higher_strike():
    low_k = call_price(100, 90, 0.05, 0.2, 1.0)
    high_k = call_price(100, 110, 0.05, 0.2, 1.0)
    assert low_k > high_k


def test_put_increases_with_higher_strike():
    low_k = put_price(100, 90, 0.05, 0.2, 1.0)
    high_k = put_price(100, 110, 0.05, 0.2, 1.0)
    assert high_k > low_k


def test_sigma_zero_call_equals_discounted_payoff_of_forward():
    """
    If sigma -> 0, S_T is deterministic: S0*exp(rT).
    Call price -> exp(-rT) * max(S0*exp(rT) - K, 0) = max(S0 - K*exp(-rT), 0).
    """
    s0, k, r, t = 100.0, 95.0, 0.05, 1.0
    # Use a tiny sigma instead of exact 0 (our formulas divide by sigma).
    c = call_price(s0, k, r, sigma=1e-8, t=t)
    expected = max(s0 - k * math.exp(-r * t), 0.0)
    assert c == pytest.approx(expected, abs=1e-4)


def test_prices_are_nonnegative():
    c = call_price(80, 100, 0.01, 0.25, 0.5)
    p = put_price(80, 100, 0.01, 0.25, 0.5)
    assert c >= 0 and p >= 0

"""Tests for Step 4. Run: python -m pytest tests/test_binomial.py -q"""

import math

import pytest

from src.binomial import crr_call, crr_put
from src.black_scholes import call_price, put_price


S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0


def test_crr_call_positive():
    assert crr_call(S0, K, R, SIGMA, T, n_steps=50) > 0


def test_crr_put_positive():
    assert crr_put(S0, K, R, SIGMA, T, n_steps=50) > 0


def test_put_call_parity_on_tree():
    n = 100
    c = crr_call(S0, K, R, SIGMA, T, n)
    p = crr_put(S0, K, R, SIGMA, T, n)
    lhs = c - p
    rhs = S0 - K * math.exp(-R * T)
    assert lhs == pytest.approx(rhs, abs=1e-8)


def test_crr_call_converges_toward_bs():
    bs = call_price(S0, K, R, SIGMA, T)
    err_coarse = abs(crr_call(S0, K, R, SIGMA, T, n_steps=25) - bs)
    err_fine = abs(crr_call(S0, K, R, SIGMA, T, n_steps=200) - bs)
    assert err_fine < err_coarse
    assert err_fine < 0.05


def test_crr_put_converges_toward_bs():
    bs = put_price(S0, K, R, SIGMA, T)
    err = abs(crr_put(S0, K, R, SIGMA, T, n_steps=300) - bs)
    assert err < 0.03


def test_higher_strike_cheaper_call_on_tree():
    assert crr_call(S0, 90, R, SIGMA, T, 80) > crr_call(S0, 110, R, SIGMA, T, 80)

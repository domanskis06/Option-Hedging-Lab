"""Tests for Step 3. Run: python -m pytest tests/test_greeks.py -q"""

import math

import pytest

from src.black_scholes import call_price, put_price
from src.greeks import (
    call_delta,
    call_rho,
    call_theta,
    gamma,
    put_delta,
    put_rho,
    put_theta,
    vega,
)


S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
EPS_S = 1e-4
EPS_SIG = 1e-6
EPS_R = 1e-6
EPS_T = 1e-6


def test_call_delta_between_0_and_1():
    d = call_delta(S0, K, R, SIGMA, T)
    assert 0.0 < d < 1.0


def test_put_delta_between_minus_1_and_0():
    d = put_delta(S0, K, R, SIGMA, T)
    assert -1.0 < d < 0.0


def test_put_delta_equals_call_delta_minus_one():
    assert put_delta(S0, K, R, SIGMA, T) == pytest.approx(
        call_delta(S0, K, R, SIGMA, T) - 1.0, abs=1e-12
    )


def test_call_delta_matches_finite_difference():
    bump = call_price(S0 + EPS_S, K, R, SIGMA, T) - call_price(S0 - EPS_S, K, R, SIGMA, T)
    fd = bump / (2 * EPS_S)
    assert call_delta(S0, K, R, SIGMA, T) == pytest.approx(fd, abs=1e-4)


def test_gamma_positive_and_same_idea_for_put_call():
    g = gamma(S0, K, R, SIGMA, T)
    assert g > 0
    # gamma from delta bump
    d_up = call_delta(S0 + EPS_S, K, R, SIGMA, T)
    d_dn = call_delta(S0 - EPS_S, K, R, SIGMA, T)
    fd = (d_up - d_dn) / (2 * EPS_S)
    assert g == pytest.approx(fd, abs=1e-4)


def test_vega_matches_finite_difference():
    bump = call_price(S0, K, R, SIGMA + EPS_SIG, T) - call_price(
        S0, K, R, SIGMA - EPS_SIG, T
    )
    fd = bump / (2 * EPS_SIG)
    assert vega(S0, K, R, SIGMA, T) == pytest.approx(fd, abs=1e-3)


def test_call_put_vega_equal():
    assert vega(S0, K, R, SIGMA, T) == pytest.approx(
        # recompute via put bump — should match call vega
        (
            put_price(S0, K, R, SIGMA + EPS_SIG, T)
            - put_price(S0, K, R, SIGMA - EPS_SIG, T)
        )
        / (2 * EPS_SIG),
        abs=1e-3,
    )


def test_call_theta_matches_finite_difference():
    # Theta = dV/dt with t = time-to-maturity shrinking, so bump T downward carefully.
    # Our analytic theta is dV/d(calendar time) ≈ -dV/dT.
    v_longer = call_price(S0, K, R, SIGMA, T + EPS_T)
    v_shorter = call_price(S0, K, R, SIGMA, T - EPS_T)
    # dV/dT ≈ (v_longer - v_shorter)/(2*EPS_T); analytic call_theta ≈ -dV/dT
    fd_dVdT = (v_longer - v_shorter) / (2 * EPS_T)
    assert call_theta(S0, K, R, SIGMA, T) == pytest.approx(-fd_dVdT, abs=1e-3)


def test_put_theta_matches_finite_difference():
    v_longer = put_price(S0, K, R, SIGMA, T + EPS_T)
    v_shorter = put_price(S0, K, R, SIGMA, T - EPS_T)
    fd_dVdT = (v_longer - v_shorter) / (2 * EPS_T)
    assert put_theta(S0, K, R, SIGMA, T) == pytest.approx(-fd_dVdT, abs=1e-3)


def test_call_rho_matches_finite_difference():
    bump = call_price(S0, K, R + EPS_R, SIGMA, T) - call_price(S0, K, R - EPS_R, SIGMA, T)
    fd = bump / (2 * EPS_R)
    assert call_rho(S0, K, R, SIGMA, T) == pytest.approx(fd, abs=1e-3)


def test_put_rho_matches_finite_difference():
    bump = put_price(S0, K, R + EPS_R, SIGMA, T) - put_price(S0, K, R - EPS_R, SIGMA, T)
    fd = bump / (2 * EPS_R)
    assert put_rho(S0, K, R, SIGMA, T) == pytest.approx(fd, abs=1e-3)


def test_known_atm_call_delta_rough():
    # For ATM-ish params, call delta should be around 0.64
    d = call_delta(100, 100, 0.05, 0.2, 1.0)
    assert 0.55 < d < 0.70

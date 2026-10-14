"""American CRR tests. Run: python -m pytest tests/test_american.py -q"""

import pytest

from src.binomial import crr_american_call, crr_american_put, crr_call, crr_put


S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
N = 200


def test_american_put_worth_at_least_european():
    am = crr_american_put(S0, K, R, SIGMA, T, N)
    eu = crr_put(S0, K, R, SIGMA, T, N)
    assert am >= eu - 1e-10


def test_american_put_has_early_exercise_premium():
    """With r>0, American put should beat European by a visible premium."""
    am = crr_american_put(S0, K, R, SIGMA, T, N)
    eu = crr_put(S0, K, R, SIGMA, T, N)
    assert am - eu > 0.05


def test_american_call_matches_european_no_dividends():
    am = crr_american_call(S0, K, R, SIGMA, T, N)
    eu = crr_call(S0, K, R, SIGMA, T, N)
    assert am == pytest.approx(eu, abs=1e-8)

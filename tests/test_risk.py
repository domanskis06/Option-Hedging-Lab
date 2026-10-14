"""Risk metric tests. Run: python -m pytest tests/test_risk.py -q"""

import numpy as np
import pytest

from src.risk import expected_shortfall, var_abs


def test_var_and_es_on_symmetric_sample():
    rng = np.random.default_rng(0)
    pnl = rng.normal(loc=0.0, scale=1.0, size=100_000)
    v = var_abs(pnl, alpha=0.95)
    es = expected_shortfall(pnl, alpha=0.95)
    # 95% VaR of N(0,1) losses ≈ 1.645
    assert v == pytest.approx(1.64485, rel=0.05)
    assert es > v


def test_worse_left_tail_increases_var():
    good = np.array([1.0, 0.5, 0.0, -0.2, -0.5])
    bad = np.array([1.0, 0.5, 0.0, -0.2, -5.0])
    assert var_abs(bad, 0.8) >= var_abs(good, 0.8)

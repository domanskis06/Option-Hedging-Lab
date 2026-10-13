"""Tests for Step 0. Run: pytest tests/test_payoffs.py -q"""

import pytest

from src.payoffs import call_payoff, put_payoff


def test_call_in_the_money():
    # Stock finished at 120, strike 100 → call pays 20
    assert call_payoff(120, 100) == 20


def test_call_out_of_the_money():
    # Stock finished at 80, strike 100 → call pays 0
    assert call_payoff(80, 100) == 0


def test_call_at_the_money():
    assert call_payoff(100, 100) == 0


def test_put_in_the_money():
    # Stock finished at 80, strike 100 → put pays 20
    assert put_payoff(80, 100) == 20


def test_put_out_of_the_money():
    assert put_payoff(120, 100) == 0


def test_put_at_the_money():
    assert put_payoff(100, 100) == 0

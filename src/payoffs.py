"""
Step 0 — Option payoffs.

YOUR JOB: implement the two functions below.
Do not use any finance libraries. Plain Python / NumPy is enough.
"""


def call_payoff(spot: float, strike: float) -> float:
    """
    European call payoff at expiry.

    Intuition (preschool version):
    - You have a coupon that lets you BUY the stock for `strike` dollars.
    - If the stock is worth MORE than `strike`, you use the coupon and
      instantly make (spot - strike) profit.
    - If the stock is worth LESS than `strike`, the coupon is useless → payoff 0.

    TODO: return max(spot - strike, 0)
    """
    raise NotImplementedError("Implement call_payoff")


def put_payoff(spot: float, strike: float) -> float:
    """
    European put payoff at expiry.

    Intuition:
    - You have a coupon that lets you SELL the stock for `strike` dollars.
    - If the stock is worth LESS than `strike`, you sell high via the coupon
      and make (strike - spot) profit.
    - If the stock is worth MORE than `strike`, the coupon is useless → payoff 0.

    TODO: return max(strike - spot, 0)
    """
    raise NotImplementedError("Implement put_payoff")

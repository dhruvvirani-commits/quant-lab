"""
tests/test_implied_vol.py
-------------------------
Verify the IV solver by ROUND-TRIP: price an option with a known sigma, then
check the solver recovers that exact sigma. If price->sigma->price is
consistent, the solver is correct.

Run from quant-lab/:  python -m pytest tests/test_implied_vol.py -v
"""
import numpy as np
import pytest
from p2_volatility.black_scholes import bs_price
from p2_volatility.implied_vol import (
    implied_vol, implied_vol_bisection, implied_vol_newton,
)


@pytest.mark.parametrize("true_sigma", [0.1, 0.2, 0.35, 0.6, 0.9, 1.5])
@pytest.mark.parametrize("ot", ["call", "put"])
def test_roundtrip_newton(true_sigma, ot):
    S, K, T, r = 100, 100, 1.0, 0.05
    price = bs_price(S, K, T, r, true_sigma, ot)
    iv = implied_vol_newton(price, S, K, T, r, ot)
    assert abs(iv - true_sigma) < 1e-4


@pytest.mark.parametrize("true_sigma", [0.15, 0.4, 0.8])
def test_roundtrip_bisection(true_sigma):
    S, K, T, r = 100, 100, 1.0, 0.05
    price = bs_price(S, K, T, r, true_sigma, "call")
    iv = implied_vol_bisection(price, S, K, T, r, "call")
    assert abs(iv - true_sigma) < 1e-4


@pytest.mark.parametrize("K", [80, 95, 100, 110, 130])
def test_roundtrip_across_strikes(K):
    """Solver must work for ITM, ATM, and OTM options."""
    S, T, r, true_sigma = 100, 0.5, 0.03, 0.45
    price = bs_price(S, K, T, r, true_sigma, "call")
    iv = implied_vol(price, S, K, T, r, "call")
    assert abs(iv - true_sigma) < 1e-3


def test_newton_and_bisection_agree():
    """Both methods should land on the same answer."""
    S, K, T, r = 100, 105, 0.75, 0.04
    price = bs_price(S, K, T, r, 0.55, "call")
    a = implied_vol_newton(price, S, K, T, r, "call")
    b = implied_vol_bisection(price, S, K, T, r, "call")
    assert abs(a - b) < 1e-4


def test_price_below_intrinsic_returns_nan():
    """No sigma can produce a price below intrinsic value -> NaN."""
    S, K, T, r = 100, 80, 1.0, 0.0
    intrinsic = S - K  # = 20
    iv = implied_vol_bisection(intrinsic - 5, S, K, T, r, "call")
    assert np.isnan(iv)

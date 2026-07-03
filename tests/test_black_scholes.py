"""
tests/test_black_scholes.py
---------------------------
Correctness tests for the BSM pricer. We check against:
  1. A known textbook value (the standard S=100,K=100,T=1,r=5%,sigma=20% case)
  2. Put-call parity (a no-arbitrage identity that MUST hold)
  3. Boundary behaviour (deep ITM/OTM, at expiry)

Run from quant-lab/:  python -m pytest tests/test_black_scholes.py -v
"""
import numpy as np
import pytest
from p2_volatility.black_scholes import bs_price, put_call_parity_gap


def test_known_textbook_call():
    """
    The canonical reference case. Every options textbook gives:
    S=100, K=100, T=1, r=0.05, sigma=0.20  ->  call ~= 10.4506
    """
    c = bs_price(100, 100, 1.0, 0.05, 0.20, "call")
    assert abs(c - 10.4506) < 0.001


def test_known_textbook_put():
    """Same inputs -> put ~= 5.5735 (from put-call parity on the above)."""
    p = bs_price(100, 100, 1.0, 0.05, 0.20, "put")
    assert abs(p - 5.5735) < 0.001


def test_put_call_parity_holds():
    """C - P must equal S - K*e^(-rT) across many random inputs."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        S = rng.uniform(50, 150)
        K = rng.uniform(50, 150)
        T = rng.uniform(0.05, 2.0)
        r = rng.uniform(0.0, 0.08)
        sigma = rng.uniform(0.1, 0.9)
        assert abs(put_call_parity_gap(S, K, T, r, sigma)) < 1e-6


def test_deep_itm_call_approaches_intrinsic():
    """
    A deep in-the-money call (spot >> strike) should be worth about its
    intrinsic value S - K*e^(-rT), since exercise is near-certain.
    """
    S, K, T, r, sigma = 200, 100, 1.0, 0.05, 0.20
    c = bs_price(S, K, T, r, sigma, "call")
    intrinsic = S - K * np.exp(-r * T)
    assert abs(c - intrinsic) < 1.0  # very close


def test_deep_otm_call_near_zero():
    """A deep out-of-the-money call (strike >> spot) is nearly worthless."""
    c = bs_price(50, 200, 0.25, 0.05, 0.20, "call")
    assert c < 0.01


def test_price_at_expiry_is_intrinsic():
    """At T=0, the option is worth exactly its intrinsic value."""
    assert bs_price(120, 100, 0, 0.05, 0.2, "call") == 20.0   # max(120-100,0)
    assert bs_price(80, 100, 0, 0.05, 0.2, "call") == 0.0     # max(80-100,0)
    assert bs_price(80, 100, 0, 0.05, 0.2, "put") == 20.0     # max(100-80,0)


def test_higher_vol_means_higher_price():
    """
    A core property: more volatility -> more option value (bigger chance of a
    big favourable move, while downside is capped). Prices must be increasing
    in sigma.
    """
    prices = [bs_price(100, 100, 1.0, 0.05, s, "call") for s in [0.1, 0.2, 0.4, 0.8]]
    assert prices == sorted(prices)  # strictly increasing


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        bs_price(100, 100, 1.0, 0.05, -0.2, "call")   # negative vol
    with pytest.raises(ValueError):
        bs_price(100, 100, 1.0, 0.05, 0.2, "banana")  # bad type

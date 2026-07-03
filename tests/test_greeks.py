"""
tests/test_greeks.py
--------------------
Verify each analytical Greek against a NUMERICAL (finite-difference) estimate.

The idea: a Greek is a derivative. So bump the relevant input by a tiny amount
h, measure how the price actually changes, and confirm the analytical Greek
matches that measured slope. If they agree, the formula is correct.

    delta ~= [Price(S+h) - Price(S-h)] / (2h)      (central difference)

Run from quant-lab/:  python -m pytest tests/test_greeks.py -v
"""
import numpy as np
import pytest
from p2_volatility.black_scholes import bs_price
from p2_volatility.greeks import delta, gamma, vega, theta, rho

# a standard test point
BASE = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.20)


def _price(S, K, T, r, sigma, ot):
    return bs_price(S, K, T, r, sigma, ot)


# ---------- DELTA: dPrice/dSpot ----------
@pytest.mark.parametrize("ot", ["call", "put"])
def test_delta_matches_numerical(ot):
    h = 0.01
    b = BASE
    num = (_price(b["S"]+h, b["K"], b["T"], b["r"], b["sigma"], ot)
           - _price(b["S"]-h, b["K"], b["T"], b["r"], b["sigma"], ot)) / (2*h)
    ana = delta(b["S"], b["K"], b["T"], b["r"], b["sigma"], ot)
    assert abs(num - ana) < 1e-4


# ---------- GAMMA: d2Price/dSpot2 ----------
def test_gamma_matches_numerical():
    h = 0.5
    b = BASE
    p_up = _price(b["S"]+h, b["K"], b["T"], b["r"], b["sigma"], "call")
    p_0  = _price(b["S"],   b["K"], b["T"], b["r"], b["sigma"], "call")
    p_dn = _price(b["S"]-h, b["K"], b["T"], b["r"], b["sigma"], "call")
    num = (p_up - 2*p_0 + p_dn) / (h**2)          # 2nd derivative
    ana = gamma(b["S"], b["K"], b["T"], b["r"], b["sigma"])
    assert abs(num - ana) < 1e-4


# ---------- VEGA: dPrice/dVol ----------
def test_vega_matches_numerical():
    h = 1e-4
    b = BASE
    num = (_price(b["S"], b["K"], b["T"], b["r"], b["sigma"]+h, "call")
           - _price(b["S"], b["K"], b["T"], b["r"], b["sigma"]-h, "call")) / (2*h)
    ana = vega(b["S"], b["K"], b["T"], b["r"], b["sigma"])
    assert abs(num - ana) < 1e-2


# ---------- THETA: dPrice/dTime (note sign) ----------
@pytest.mark.parametrize("ot", ["call", "put"])
def test_theta_matches_numerical(ot):
    # CONVENTION: theta is the price change as TIME PASSES (T decreases).
    # The finite difference dP/dT measures change as T INCREASES, so theta is
    # the NEGATIVE of dP/dT. We verify: theta ~= -(dP/dT).
    h = 1e-4
    b = BASE
    dP_dT = (_price(b["S"], b["K"], b["T"]+h, b["r"], b["sigma"], ot)
             - _price(b["S"], b["K"], b["T"]-h, b["r"], b["sigma"], ot)) / (2*h)
    ana = theta(b["S"], b["K"], b["T"], b["r"], b["sigma"], ot)
    assert abs((-dP_dT) - ana) < 1e-1


# ---------- RHO: dPrice/dRate ----------
@pytest.mark.parametrize("ot", ["call", "put"])
def test_rho_matches_numerical(ot):
    h = 1e-5
    b = BASE
    num = (_price(b["S"], b["K"], b["T"], b["r"]+h, b["sigma"], ot)
           - _price(b["S"], b["K"], b["T"], b["r"]-h, b["sigma"], ot)) / (2*h)
    ana = rho(b["S"], b["K"], b["T"], b["r"], b["sigma"], ot)
    assert abs(num - ana) < 1e-1


# ---------- property checks ----------
def test_delta_bounds():
    """Call delta in [0,1], put delta in [-1,0]."""
    assert 0 <= delta(100, 100, 1, 0.05, 0.2, "call") <= 1
    assert -1 <= delta(100, 100, 1, 0.05, 0.2, "put") <= 0


def test_gamma_vega_positive():
    """Gamma and vega are always positive (same for calls and puts)."""
    assert gamma(100, 100, 1, 0.05, 0.2) > 0
    assert vega(100, 100, 1, 0.05, 0.2) > 0


def test_atm_call_delta_near_half():
    """A short-dated at-the-money call has delta near 0.5."""
    d = delta(100, 100, 0.02, 0.05, 0.2, "call")
    assert 0.45 < d < 0.60


def test_call_put_delta_relationship():
    """No-arbitrage: call_delta - put_delta = 1 exactly."""
    cd = delta(100, 105, 0.7, 0.03, 0.3, "call")
    pd = delta(100, 105, 0.7, 0.03, 0.3, "put")
    assert abs((cd - pd) - 1.0) < 1e-9

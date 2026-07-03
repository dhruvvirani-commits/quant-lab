"""
p2_volatility/black_scholes.py
------------------------------
The Black-Scholes-Merton (BSM) option pricer, built from scratch.

This is the ENGINE of Project 2. Everything else — the Greeks, implied
volatility, the vol surface — plugs into this. We build it first, verify it
against known values, then extend.

============================================================================
 CONCEPT CHECK — Black-Scholes  (full detail: Part 3 of your companion doc)
============================================================================
An option gives the right (not obligation) to buy (call) or sell (put) an
asset at a fixed strike price K, by expiry T. BSM computes the FAIR PREMIUM
you should pay for that right, from 5 inputs:

    S     spot price (current asset price)
    K     strike price (the fixed price in the contract)
    T     time to expiry, in YEARS (30 days = 30/365 = 0.0822)
    r     risk-free interest rate (e.g. 0.05 for 5%)
    sigma volatility  <-- the only input you can't directly observe

The call formula:   C = S*N(d1) - K*e^(-rT)*N(d2)
    - S*N(d1)          : expected value of receiving the asset
    - K*e^(-rT)*N(d2)  : expected cost of paying the strike, discounted to today
    - the price is (what you expect to get) minus (what you expect to pay)

N(.) is the standard normal CDF: it turns a number into a probability 0..1,
loosely "the chance of finishing in-the-money."
============================================================================
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm  # norm.cdf is our N(.)


def _d1_d2(S, K, T, r, sigma):
    """
    The two intermediate terms d1 and d2 that appear in every BSM formula.

        d1 = [ ln(S/K) + (r + sigma^2/2) * T ] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

    Kept in one helper because the price AND all the Greeks reuse them.
    """
    # guard against degenerate inputs that would divide by zero
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive.")
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type: str = "call") -> float:
    """
    Black-Scholes fair price of a European option.

    Parameters
    ----------
    S, K, T, r, sigma : the five inputs (see module docstring)
    option_type       : "call" or "put"

    Returns
    -------
    The fair premium (a positive number).

    At/after expiry (T<=0) we return the intrinsic value: what the option is
    worth if exercised right now, i.e. max(S-K, 0) for a call.
    """
    option_type = option_type.lower()

    # handle expiry cleanly: value = intrinsic value, no time value left
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        elif option_type == "put":
            return max(K - S, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'.")

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    disc = np.exp(-r * T)  # e^(-rT): discount factor, "present value of $1 at T"

    if option_type == "call":
        # C = S*N(d1) - K*e^(-rT)*N(d2)
        return float(S * norm.cdf(d1) - K * disc * norm.cdf(d2))
    elif option_type == "put":
        # P = K*e^(-rT)*N(-d2) - S*N(-d1)
        return float(K * disc * norm.cdf(-d2) - S * norm.cdf(-d1))
    else:
        raise ValueError("option_type must be 'call' or 'put'.")


# ---------------------------------------------------------------------------
# PUT-CALL PARITY  — a free correctness check we can test against.
# ---------------------------------------------------------------------------
# CONCEPT CHECK — Put-Call Parity:
# A fundamental no-arbitrage relationship linking calls and puts:
#
#       C - P = S - K*e^(-rT)
#
# If this doesn't hold, a risk-free arbitrage would exist. So we can use it to
# verify our pricer: price a call and a put, and check the identity holds.
def put_call_parity_gap(S, K, T, r, sigma) -> float:
    """Returns C - P - (S - K*e^(-rT)). Should be ~0 for correct pricing."""
    c = bs_price(S, K, T, r, sigma, "call")
    p = bs_price(S, K, T, r, sigma, "put")
    return c - p - (S - K * np.exp(-r * T))


if __name__ == "__main__":
    # quick sanity demo: an at-the-money BTC-style call
    S, K, T, r, sigma = 68000, 70000, 30/365, 0.05, 0.60
    c = bs_price(S, K, T, r, sigma, "call")
    p = bs_price(S, K, T, r, sigma, "put")
    print(f"Call price: {c:,.2f}")
    print(f"Put  price: {p:,.2f}")
    print(f"Put-call parity gap (should be ~0): {put_call_parity_gap(S,K,T,r,sigma):.6f}")

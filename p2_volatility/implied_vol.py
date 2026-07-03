"""
p2_volatility/implied_vol.py
----------------------------
Solve for implied volatility: the sigma that makes Black-Scholes reproduce a
given market price.

============================================================================
 CONCEPT CHECK — Implied Volatility  (full detail: Part 5 of your doc)
============================================================================
In the market we can SEE the option's price, and we know S, K, T, r. The only
unknown is volatility. So we invert the problem: find the sigma that makes our
BSM pricer output exactly the observed market price. That sigma is the IMPLIED
VOLATILITY -- the market's collective guess about future volatility.

sigma can't be isolated algebraically (it's buried inside N(d1), d1, d2), so we
solve NUMERICALLY. Two methods:
  - Bisection      : bracket the answer, halve repeatedly. Slow but bulletproof.
  - Newton-Raphson : use vega (dPrice/dVol) to jump toward the answer. Fast.
This is where VEGA (Step 2) pays off -- it powers the fast solver.
============================================================================
"""

from __future__ import annotations
import numpy as np
from .black_scholes import bs_price
from .greeks import vega


def implied_vol_bisection(price, S, K, T, r, option_type="call",
                          lo=1e-4, hi=5.0, tol=1e-6, max_iter=200):
    """
    Find implied vol by bisection.

    We know BSM price is INCREASING in sigma (more vol -> more value). So if the
    target price sits between price(lo) and price(hi), we can repeatedly halve
    the [lo, hi] interval, always keeping the target bracketed, until we've
    pinned sigma to within `tol`.

    Returns np.nan if the price is outside the achievable range (e.g. below
    intrinsic value -- no real sigma can produce it).
    """
    p_lo = bs_price(S, K, T, r, lo, option_type)
    p_hi = bs_price(S, K, T, r, hi, option_type)
    # target must be inside [p_lo, p_hi] for a solution to exist
    if not (p_lo <= price <= p_hi):
        return np.nan

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = bs_price(S, K, T, r, mid, option_type)
        if abs(p_mid - price) < tol:
            return mid
        # keep the half of the interval that still brackets the target
        if p_mid < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def implied_vol_newton(price, S, K, T, r, option_type="call",
                       guess=0.5, tol=1e-6, max_iter=100):
    """
    Find implied vol by Newton-Raphson -- much faster than bisection.

    Newton's update:  sigma_new = sigma - f(sigma)/f'(sigma)
    where f(sigma) = BSM_price(sigma) - market_price, and f'(sigma) = vega.

    Falls back to bisection if Newton misbehaves (vega ~ 0, or it steps out of
    a sensible range). This hybrid is what production code does: fast when it
    can be, robust always.
    """
    sigma = guess
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, option_type) - price
        if abs(diff) < tol:
            return sigma
        v = vega(S, K, T, r, sigma)
        if v < 1e-8:            # vega too small -> Newton unstable, bail out
            break
        sigma = sigma - diff / v
        if sigma <= 0 or sigma > 10:   # stepped somewhere silly
            break
    # robust fallback
    return implied_vol_bisection(price, S, K, T, r, option_type)


def implied_vol(price, S, K, T, r, option_type="call"):
    """Public entry point: fast Newton with a bisection safety net."""
    return implied_vol_newton(price, S, K, T, r, option_type)


if __name__ == "__main__":
    # Round-trip demo: price with a KNOWN vol, then recover it.
    S, K, T, r, true_sigma = 100, 100, 1.0, 0.05, 0.35
    market_price = bs_price(S, K, T, r, true_sigma, "call")
    iv_newton = implied_vol_newton(market_price, S, K, T, r, "call")
    iv_bisect = implied_vol_bisection(market_price, S, K, T, r, "call")
    print(f"True sigma       : {true_sigma}")
    print(f"Recovered (Newton): {iv_newton:.6f}")
    print(f"Recovered (Bisect): {iv_bisect:.6f}")

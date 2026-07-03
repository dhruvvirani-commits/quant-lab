"""
p2_volatility/greeks.py
-----------------------
The five option Greeks, built from scratch. Each Greek measures the option's
sensitivity to ONE input. Together they are the risk dashboard of options.

============================================================================
 CONCEPT CHECK — The Greeks  (full detail: Part 4 of your companion doc)
============================================================================
Think of driving a car:
    Delta (Δ) = speed         -> how fast option value moves when spot moves $1
    Gamma (Γ) = acceleration  -> how fast delta itself changes
    Vega  (ν) = road roughness-> sensitivity to a 1% change in volatility
    Theta (Θ) = fuel burning  -> value lost per day just from time passing
    Rho   (ρ) = rates         -> sensitivity to interest rates (usually minor)

Each Greek is a partial derivative of the Black-Scholes price:
    Delta = dPrice/dSpot,  Vega = dPrice/dVol,  Theta = dPrice/dTime, etc.
They all reuse the same d1, d2 terms from the pricer.
============================================================================
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm
from .black_scholes import _d1_d2

# norm.pdf is the standard normal PDF (the bell curve height), written n(.)
# norm.cdf is the standard normal CDF, written N(.)


def delta(S, K, T, r, sigma, option_type: str = "call") -> float:
    """
    Δ = dPrice / dSpot.  "If spot moves $1, the option moves ~$Δ."
      call delta = N(d1)      -> between 0 and 1
      put  delta = N(d1) - 1  -> between -1 and 0
    An at-the-money option has delta ~0.5: it moves about half as much as spot.
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type.lower() == "call":
        return float(norm.cdf(d1))
    elif option_type.lower() == "put":
        return float(norm.cdf(d1) - 1.0)
    raise ValueError("option_type must be 'call' or 'put'.")


def gamma(S, K, T, r, sigma) -> float:
    """
    Γ = dDelta / dSpot = d²Price / dSpot².  Same for calls and puts.
    Highest for at-the-money options near expiry (delta swings fastest there).
      gamma = n(d1) / (S * sigma * sqrt(T))
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def vega(S, K, T, r, sigma) -> float:
    """
    ν = dPrice / dVol.  Same for calls and puts. Always positive for a buyer:
    more volatility = more option value. Reported PER 1% (0.01) change in vol.
      vega = S * n(d1) * sqrt(T)
    NOTE: this returns vega per 1.00 (100%) change in vol; divide by 100 for
    "per 1% point". We keep the raw form here; the IV solver (Step 3) uses it.
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return float(S * norm.pdf(d1) * np.sqrt(T))


def theta(S, K, T, r, sigma, option_type: str = "call") -> float:
    """
    Θ = dPrice / dTime.  Almost always negative for a buyer: the option loses
    value as expiry approaches ("time decay"). Reported per YEAR here; divide
    by 365 for per-day.
      call theta = -S*n(d1)*sigma/(2*sqrt(T)) - r*K*e^(-rT)*N(d2)
      put  theta = -S*n(d1)*sigma/(2*sqrt(T)) + r*K*e^(-rT)*N(-d2)
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    disc = np.exp(-r * T)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type.lower() == "call":
        return float(term1 - r * K * disc * norm.cdf(d2))
    elif option_type.lower() == "put":
        return float(term1 + r * K * disc * norm.cdf(-d2))
    raise ValueError("option_type must be 'call' or 'put'.")


def rho(S, K, T, r, sigma, option_type: str = "call") -> float:
    """
    ρ = dPrice / dRate.  Usually the least important Greek, especially for
    short-dated crypto options. Reported per 1.00 change in r.
      call rho =  K*T*e^(-rT)*N(d2)
      put  rho = -K*T*e^(-rT)*N(-d2)
    """
    _, d2 = _d1_d2(S, K, T, r, sigma)
    disc = np.exp(-r * T)
    if option_type.lower() == "call":
        return float(K * T * disc * norm.cdf(d2))
    elif option_type.lower() == "put":
        return float(-K * T * disc * norm.cdf(-d2))
    raise ValueError("option_type must be 'call' or 'put'.")


def all_greeks(S, K, T, r, sigma, option_type: str = "call") -> dict:
    """Convenience: return every Greek in one dict, in practical units."""
    return {
        "delta": delta(S, K, T, r, sigma, option_type),
        "gamma": gamma(S, K, T, r, sigma),
        "vega_per_1pct": vega(S, K, T, r, sigma) / 100.0,   # per 1% vol move
        "theta_per_day": theta(S, K, T, r, sigma, option_type) / 365.0,
        "rho_per_1pct": rho(S, K, T, r, sigma, option_type) / 100.0,
    }


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    print("At-the-money call Greeks (S=K=100, T=1, r=5%, vol=20%):")
    for k, v in all_greeks(S, K, T, r, sigma, "call").items():
        print(f"  {k:<16} {v:+.4f}")

"""
p2_volatility/realized_vol.py
-----------------------------
Compute realized volatility from price history and compare it to implied vol.

============================================================================
 CONCEPT CHECK — Implied vs Realized  (full detail: Part 7 of your doc)
============================================================================
Two kinds of volatility:
  IMPLIED  = the market's EXPECTATION of future vol, from option prices. Forward.
  REALIZED = the vol that ACTUALLY happened, from historical returns. Backward.

Empirically, implied tends to sit ABOVE realized -- option buyers overpay for
protection, sellers collect the difference. That gap is the VOLATILITY RISK
PREMIUM, one of the most robust edges in finance. It connects to P1's lesson:
a real edge names who's on the other side (insurance buyers) and why they lose
(they pay for peace of mind), just like insurance.
============================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def realized_volatility(prices: pd.Series, window: int = 30,
                        periods_per_year: int = 365) -> pd.Series:
    """
    Annualized rolling realized volatility from a price series.

    Method: take log returns, compute their rolling standard deviation, then
    annualize by multiplying by sqrt(periods_per_year).

        realized_vol = std(log_returns) * sqrt(periods_per_year)

    `periods_per_year` = 365 for daily crypto (trades every day), 252 for
    daily equities (trading days only).
    """
    log_ret = np.log(prices / prices.shift(1))
    roll_std = log_ret.rolling(window).std()
    return roll_std * np.sqrt(periods_per_year)


def realized_vol_close_to_close(prices: pd.Series, window: int = 30,
                                periods_per_year: int = 365) -> float:
    """Single most-recent realized-vol number (scalar), for quick comparison."""
    rv = realized_volatility(prices, window, periods_per_year)
    return float(rv.dropna().iloc[-1]) if rv.notna().any() else np.nan


def vol_risk_premium(implied: float, realized: float) -> float:
    """
    The gap: implied minus realized. Positive means the market is pricing in
    MORE vol than actually happened -- the premium option sellers earn.
    """
    return implied - realized


def make_synthetic_prices(n: int = 400, true_vol: float = 0.60,
                          seed: int = 5, start: float = 68000.0) -> pd.Series:
    """
    Synthetic daily price path with a KNOWN annualized volatility, so we can
    verify realized_volatility() recovers roughly `true_vol`.
    """
    rng = np.random.default_rng(seed)
    daily_vol = true_vol / np.sqrt(365)
    rets = rng.normal(0, daily_vol, n)
    prices = start * np.exp(np.cumsum(rets))
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(prices, index=idx)


if __name__ == "__main__":
    prices = make_synthetic_prices(true_vol=0.60)
    rv = realized_vol_close_to_close(prices, window=30)
    print(f"Imposed vol : 0.60")
    print(f"Realized vol: {rv:.3f}  (should be near 0.60)")
    # example premium
    implied = 0.68
    print(f"If implied = {implied}, vol risk premium = "
          f"{vol_risk_premium(implied, rv):+.3f}")

"""
p3_risk/returns.py
------------------
The foundation of portfolio risk: turning prices into returns, and building a
multi-asset returns table.

============================================================================
 CONCEPT CHECK — Returns  (the raw material of all risk analysis)
============================================================================
Risk is about how much things MOVE, not their price level. So we never work
with prices directly -- we work with RETURNS (percentage changes).

Two kinds:
  - Simple return:  (P_today - P_yesterday) / P_yesterday
  - Log return:     ln(P_today / P_yesterday)   <- we use these

Why log returns? They add up nicely over time and behave better statistically.
For small daily moves, log and simple returns are almost identical.

A portfolio's return is just the weighted average of its assets' returns:
    portfolio_return = w1*r1 + w2*r2 + ... (weights sum to 1)
That simple fact is the whole basis of portfolio math.
============================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def to_log_returns(prices: pd.Series | pd.DataFrame):
    """Convert a price series (or table of prices) to daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def build_returns_table(price_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Given {asset_name: price_series}, align them on common dates and return a
    single DataFrame of daily log returns (one column per asset).

    Alignment matters: assets trade on different calendars (crypto = 7 days,
    equities = weekdays). We keep only dates where ALL assets have data, so
    correlations and portfolio math are computed on a consistent sample.
    """
    prices = pd.DataFrame(price_dict).dropna()
    return to_log_returns(prices)


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    Daily return of a portfolio = weighted sum of asset returns.
    weights must sum to 1 and match the column order of `returns`.
    """
    weights = np.asarray(weights, dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must sum to 1")
    if len(weights) != returns.shape[1]:
        raise ValueError("weights length must match number of assets")
    return returns.dot(weights)


def annualize_return(daily_returns: pd.Series, periods_per_year: int = 365) -> float:
    """Compound daily returns into an annualized figure."""
    total = np.exp(daily_returns.sum()) - 1        # log returns -> total return
    years = len(daily_returns) / periods_per_year
    return (1 + total) ** (1 / years) - 1 if years > 0 else 0.0


def annualize_vol(daily_returns: pd.Series, periods_per_year: int = 365) -> float:
    """
    Annualized volatility = daily std * sqrt(periods).
    CONCEPT: volatility scales with the SQUARE ROOT of time, not linearly.
    This is why a daily vol of 3% becomes ~57% annualized (0.03*sqrt(365)).
    """
    return daily_returns.std() * np.sqrt(periods_per_year)


# ---------------------------------------------------------------------------
# synthetic multi-asset data for testing without network
# ---------------------------------------------------------------------------
def make_synthetic_prices(assets=("BTC", "ETH", "GOLD", "SPX"),
                          n: int = 800, seed: int = 11) -> dict[str, pd.Series]:
    """
    Build correlated synthetic price paths with realistic properties:
    crypto is volatile and correlated with each other; gold is calm and
    slightly negatively correlated with equities. Enough structure to test
    correlations and portfolio math.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")

    # a shared "market" factor plus asset-specific noise creates correlation
    market = rng.normal(0, 0.015, n)
    profiles = {
        "BTC":  (0.0004, 0.035, 1.2),   # (drift, own-vol, market-beta)
        "ETH":  (0.0004, 0.040, 1.3),
        "GOLD": (0.0002, 0.010, -0.2),
        "SPX":  (0.0003, 0.012, 0.8),
        "DXY":  (0.0000, 0.006, -0.3),
    }
    out = {}
    for a in assets:
        drift, vol, beta = profiles.get(a, (0.0002, 0.02, 0.5))
        rets = drift + beta * market + rng.normal(0, vol, n)
        out[a] = pd.Series(100 * np.exp(np.cumsum(rets)), index=idx, name=a)
    return out


if __name__ == "__main__":
    prices = make_synthetic_prices()
    rets = build_returns_table(prices)
    print("Assets:", list(rets.columns))
    print("Days:", len(rets))
    print("\nAnnualized vol per asset:")
    for a in rets.columns:
        print(f"  {a:<5} {annualize_vol(rets[a])*100:.1f}%")

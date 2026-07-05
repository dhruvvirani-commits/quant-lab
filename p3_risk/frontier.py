"""
p3_risk/frontier.py
-------------------
Markowitz portfolio optimization and the efficient frontier.

============================================================================
 CONCEPT CHECK — The Efficient Frontier  (Nobel-Prize portfolio theory)
============================================================================
Given a set of assets, there are infinitely many ways to weight them. Each mix
gives some expected return and some risk (volatility). If you plot every
possible mix on a risk (x) vs return (y) chart, they form a cloud.

The EFFICIENT FRONTIER is the upper-left edge of that cloud: the portfolios
that give the MAXIMUM return for each level of risk (or minimum risk for each
level of return). Any portfolio below the frontier is "inefficient" -- you
could get more return for the same risk.

Two special points on it:
  - Minimum-variance portfolio: the lowest-risk mix possible.
  - Maximum-Sharpe portfolio (the "tangency" portfolio): the best
    risk-ADJUSTED return -- the highest reward per unit of risk.

THE HONEST CAVEAT: Markowitz assumes you know future returns and correlations.
You don't -- you estimate them from history, and they're unstable. So the
"optimal" weights are fragile. Understanding this limitation is as important
as the math. Real desks use it as a guide, not gospel.
============================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _portfolio_stats(weights, mean_returns, cov, periods_per_year=365):
    """Annualized (return, volatility, Sharpe) for a given weight vector."""
    ann_ret = np.dot(weights, mean_returns) * periods_per_year
    ann_vol = np.sqrt(weights @ cov @ weights) * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    return ann_ret, ann_vol, sharpe


def random_portfolios(returns: pd.DataFrame, n_portfolios: int = 20_000,
                      seed: int = 3, periods_per_year: int = 365) -> pd.DataFrame:
    """
    Generate many random portfolios to trace out the risk/return cloud.
    Each row: the weights plus the resulting return, vol, and Sharpe.
    This is the simple, transparent way to visualize the frontier.
    """
    rng = np.random.default_rng(seed)
    mean_returns = returns.mean().values
    cov = returns.cov().values
    n_assets = returns.shape[1]

    rows = []
    for _ in range(n_portfolios):
        w = rng.random(n_assets)
        w /= w.sum()                      # weights sum to 1, no shorting
        ret, vol, sharpe = _portfolio_stats(w, mean_returns, cov, periods_per_year)
        rows.append({"return": ret, "vol": vol, "sharpe": sharpe,
                     **{returns.columns[i]: w[i] for i in range(n_assets)}})
    return pd.DataFrame(rows)


def min_variance_portfolio(returns: pd.DataFrame, periods_per_year: int = 365):
    """
    The lowest-risk portfolio, found analytically from the covariance matrix:
        w = (inv(cov) @ ones) / (ones @ inv(cov) @ ones)
    """
    cov = returns.cov().values
    ones = np.ones(cov.shape[0])
    inv = np.linalg.inv(cov)
    w = inv @ ones / (ones @ inv @ ones)
    ret, vol, sharpe = _portfolio_stats(w, returns.mean().values, cov, periods_per_year)
    return {"weights": dict(zip(returns.columns, w)),
            "return": ret, "vol": vol, "sharpe": sharpe}


def max_sharpe_portfolio(cloud: pd.DataFrame, returns: pd.DataFrame):
    """The portfolio with the best risk-adjusted return, from the random cloud."""
    best = cloud.loc[cloud["sharpe"].idxmax()]
    weights = {a: best[a] for a in returns.columns}
    return {"weights": weights, "return": best["return"],
            "vol": best["vol"], "sharpe": best["sharpe"]}


if __name__ == "__main__":
    from p3_risk.returns import make_synthetic_prices, build_returns_table
    rets = build_returns_table(make_synthetic_prices())
    cloud = random_portfolios(rets, n_portfolios=5000)
    mv = min_variance_portfolio(rets)
    ms = max_sharpe_portfolio(cloud, rets)
    print("Minimum-variance portfolio:")
    print(f"  vol {mv['vol']*100:.1f}%  return {mv['return']*100:.1f}%  Sharpe {mv['sharpe']:.2f}")
    print(f"  weights: " + ", ".join(f"{k} {v*100:.0f}%" for k,v in mv['weights'].items()))
    print("\nMax-Sharpe portfolio:")
    print(f"  vol {ms['vol']*100:.1f}%  return {ms['return']*100:.1f}%  Sharpe {ms['sharpe']:.2f}")
    print(f"  weights: " + ", ".join(f"{k} {v*100:.0f}%" for k,v in ms['weights'].items()))

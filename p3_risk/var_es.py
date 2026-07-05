"""
p3_risk/var_es.py
-----------------
Value-at-Risk (VaR) and Expected Shortfall (ES) -- the two core measures of
"how much could I lose?"

============================================================================
 CONCEPT CHECK — Value-at-Risk  (the industry's main risk number)
============================================================================
VaR answers: "On a bad day, how much could I lose?"

More precisely, 95% VaR = the loss you expect to EXCEED only 5% of the time.
If your 1-day 95% VaR is $10,000, then on 95% of days you lose less than that,
and on the worst 5% of days you lose MORE.

Three ways to compute it (they disagree, which is the interesting part):
  1. Historical  -- just look at past returns, take the 5th percentile. No
                    assumptions. But assumes the future looks like the past.
  2. Parametric  -- assume returns are normally distributed, use mean & std.
                    Clean, but the normal curve UNDERSTATES crashes (fat tails).
  3. Monte-Carlo -- simulate thousands of random future days from the stats.
                    Flexible, good for complex portfolios.

============================================================================
 CONCEPT CHECK — Expected Shortfall  (the measure that fixed VaR's blind spot)
============================================================================
VaR tells you the THRESHOLD of a bad day, but not how bad it gets BEYOND it.
Two portfolios can have the same VaR but wildly different tail risk.

Expected Shortfall (ES, also called CVaR) answers: "WHEN you breach the VaR,
what's your AVERAGE loss?" It's the mean of the worst 5% of outcomes. Because
it looks at the whole tail, regulators (Basel) moved from VaR to ES. It's the
more honest number.
============================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# VALUE-AT-RISK  (returned as a POSITIVE number = the size of the loss)
# ---------------------------------------------------------------------------
def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR: the empirical percentile of past losses. No distribution
    assumption -- we just sort the returns and find the cutoff.
    """
    alpha = 1 - confidence
    return -np.percentile(returns.dropna(), alpha * 100)


def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Parametric (Gaussian) VaR: assume normal returns. VaR = -(mu + z*sigma),
    where z is the normal quantile. Fast, but underestimates tail risk because
    real returns have fatter tails than the normal distribution.
    """
    mu, sigma = returns.mean(), returns.std()
    z = norm.ppf(1 - confidence)          # negative number (left tail)
    return -(mu + z * sigma)


def var_monte_carlo(returns: pd.Series, confidence: float = 0.95,
                    n_sims: int = 50_000, seed: int = 0) -> float:
    """
    Monte-Carlo VaR: fit mean & std, simulate many random days, take the
    percentile of the simulated losses. Here we simulate from a normal, but
    the framework extends to any distribution.
    """
    rng = np.random.default_rng(seed)
    mu, sigma = returns.mean(), returns.std()
    sims = rng.normal(mu, sigma, n_sims)
    alpha = 1 - confidence
    return -np.percentile(sims, alpha * 100)


# ---------------------------------------------------------------------------
# EXPECTED SHORTFALL  (average loss in the tail beyond VaR)
# ---------------------------------------------------------------------------
def expected_shortfall_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical ES: the AVERAGE of all returns worse than the VaR threshold.
    This is the mean of the tail -- the 'when it's bad, how bad' number.
    """
    r = returns.dropna()
    var = var_historical(r, confidence)
    tail = r[r <= -var]                    # the losses beyond VaR
    return -tail.mean() if len(tail) else var


def expected_shortfall_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Parametric ES under a normal: closed-form using the pdf at the quantile.
    ES = -(mu - sigma * pdf(z)/(1-confidence)).
    """
    mu, sigma = returns.mean(), returns.std()
    alpha = 1 - confidence
    z = norm.ppf(alpha)
    return -(mu - sigma * norm.pdf(z) / alpha)


def risk_report(returns: pd.Series, confidence: float = 0.95,
                capital: float = 10_000.0) -> dict:
    """One dict with every VaR/ES measure, in both % and dollar terms."""
    vh = var_historical(returns, confidence)
    vp = var_parametric(returns, confidence)
    vm = var_monte_carlo(returns, confidence)
    eh = expected_shortfall_historical(returns, confidence)
    ep = expected_shortfall_parametric(returns, confidence)
    return {
        "confidence": confidence,
        "var_historical_pct": vh * 100,
        "var_parametric_pct": vp * 100,
        "var_monte_carlo_pct": vm * 100,
        "es_historical_pct": eh * 100,
        "es_parametric_pct": ep * 100,
        "var_historical_$": vh * capital,
        "es_historical_$": eh * capital,
    }


if __name__ == "__main__":
    from p3_risk.returns import make_synthetic_prices, build_returns_table, portfolio_returns
    import numpy as np
    rets = build_returns_table(make_synthetic_prices())
    port = portfolio_returns(rets, np.array([0.25, 0.25, 0.25, 0.25]))
    rep = risk_report(port, capital=10_000)
    print("Equal-weight portfolio, 95% 1-day risk on $10,000:\n")
    print(f"  VaR (historical) : {rep['var_historical_pct']:.2f}%  = ${rep['var_historical_$']:,.0f}")
    print(f"  VaR (parametric) : {rep['var_parametric_pct']:.2f}%")
    print(f"  VaR (monte-carlo): {rep['var_monte_carlo_pct']:.2f}%")
    print(f"  ES  (historical) : {rep['es_historical_pct']:.2f}%  = ${rep['es_historical_$']:,.0f}")
    print(f"\n  Note: ES > VaR always -- the tail is worse than the threshold.")

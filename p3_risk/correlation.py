"""
p3_risk/correlation.py
----------------------
Correlation and diversification analysis.

============================================================================
 CONCEPT CHECK — Correlation  (the math of diversification)
============================================================================
Correlation measures how two assets move TOGETHER, on a scale of -1 to +1:
  +1  = they move in lockstep (no diversification benefit)
   0  = they move independently (good diversification)
  -1  = they move oppositely (one hedges the other)

Diversification works because combining assets that DON'T move together
reduces total portfolio risk without necessarily reducing return. Two 20%-vol
assets that are uncorrelated combine into a portfolio with LESS than 20% vol.
That "free" risk reduction is the only free lunch in finance.

THE CATCH (this is the important, non-obvious part):
Correlations are NOT stable. In calm markets, assets look diversified. In a
CRASH, correlations spike toward +1 -- everything falls together, exactly when
you needed diversification most. We measure ROLLING correlation to see this.
============================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Static correlation matrix across all assets (the full-sample picture)."""
    return returns.corr()


def rolling_correlation(returns: pd.DataFrame, asset_a: str, asset_b: str,
                        window: int = 60) -> pd.Series:
    """
    Rolling correlation between two assets over time. This reveals CORRELATION
    INSTABILITY -- how the relationship shifts, especially spiking during
    stress. A flat correlation is a myth; this shows the truth.
    """
    return returns[asset_a].rolling(window).corr(returns[asset_b])


def diversification_ratio(returns: pd.DataFrame, weights: np.ndarray) -> float:
    """
    Diversification ratio = (weighted avg of individual vols) / (portfolio vol).
    A value > 1 means diversification is working -- the portfolio is less
    volatile than the sum of its parts. Higher = more diversification benefit.
    """
    weights = np.asarray(weights, dtype=float)
    individual_vols = returns.std().values
    weighted_avg_vol = np.dot(weights, individual_vols)
    port_vol = np.sqrt(weights @ returns.cov().values @ weights)
    return weighted_avg_vol / port_vol if port_vol > 0 else np.nan


def average_correlation(returns: pd.DataFrame) -> float:
    """Average pairwise correlation across the portfolio (a single summary number)."""
    corr = returns.corr().values
    n = corr.shape[0]
    # average of the off-diagonal entries
    off_diag = corr[~np.eye(n, dtype=bool)]
    return float(off_diag.mean())


if __name__ == "__main__":
    from p3_risk.returns import make_synthetic_prices, build_returns_table
    import numpy as np
    rets = build_returns_table(make_synthetic_prices())
    print("Correlation matrix:")
    print(correlation_matrix(rets).round(2).to_string())
    print(f"\nAverage pairwise correlation: {average_correlation(rets):.2f}")
    w = np.array([0.25, 0.25, 0.25, 0.25])
    print(f"Diversification ratio (equal weight): {diversification_ratio(rets, w):.2f}")

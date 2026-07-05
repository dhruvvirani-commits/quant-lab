"""
tests/test_risk.py
------------------
Correctness tests for P3 risk analytics: returns, VaR/ES, correlation, frontier.
"""
import numpy as np
import pandas as pd
import pytest

from p3_risk.returns import (
    to_log_returns, build_returns_table, portfolio_returns,
    annualize_vol, make_synthetic_prices,
)
from p3_risk.var_es import (
    var_historical, var_parametric, var_monte_carlo,
    expected_shortfall_historical, expected_shortfall_parametric,
)
from p3_risk.correlation import (
    correlation_matrix, diversification_ratio, average_correlation,
)
from p3_risk.frontier import (
    random_portfolios, min_variance_portfolio, max_sharpe_portfolio,
)


@pytest.fixture
def rets():
    return build_returns_table(make_synthetic_prices(seed=1))


# ---------- returns ----------
def test_portfolio_weights_must_sum_to_one(rets):
    with pytest.raises(ValueError):
        portfolio_returns(rets, np.array([0.5, 0.5, 0.5, 0.5]))


def test_portfolio_return_is_weighted_average(rets):
    w = np.array([0.25, 0.25, 0.25, 0.25])
    port = portfolio_returns(rets, w)
    # equal-weight portfolio return equals mean across columns each day
    expected = rets.mean(axis=1)
    assert np.allclose(port.values, expected.values)


# ---------- VaR / ES ----------
def test_var_is_positive(rets):
    port = portfolio_returns(rets, np.array([0.25]*4))
    assert var_historical(port) > 0
    assert var_parametric(port) > 0


def test_es_greater_than_var(rets):
    """Expected Shortfall must always exceed VaR (tail is worse than threshold)."""
    port = portfolio_returns(rets, np.array([0.25]*4))
    assert expected_shortfall_historical(port) >= var_historical(port)
    assert expected_shortfall_parametric(port) >= var_parametric(port)


def test_higher_confidence_higher_var(rets):
    """99% VaR must be larger than 95% VaR (further into the tail)."""
    port = portfolio_returns(rets, np.array([0.25]*4))
    assert var_historical(port, 0.99) > var_historical(port, 0.95)


def test_var_methods_roughly_agree(rets):
    """On near-normal data, the three VaR methods should be close."""
    port = portfolio_returns(rets, np.array([0.25]*4))
    vh = var_historical(port)
    vp = var_parametric(port)
    vm = var_monte_carlo(port)
    assert abs(vh - vp) < 0.01 and abs(vp - vm) < 0.01


# ---------- correlation ----------
def test_correlation_matrix_diagonal_is_one(rets):
    corr = correlation_matrix(rets)
    assert np.allclose(np.diag(corr.values), 1.0)


def test_correlation_bounded(rets):
    corr = correlation_matrix(rets).values
    assert (corr >= -1.001).all() and (corr <= 1.001).all()


def test_diversification_ratio_above_one(rets):
    """A diversified portfolio has ratio > 1 (less vol than sum of parts)."""
    w = np.array([0.25]*4)
    assert diversification_ratio(rets, w) > 1.0


# ---------- frontier ----------
def test_min_variance_has_lowest_vol(rets):
    """The min-variance portfolio should have lower vol than most random ones."""
    cloud = random_portfolios(rets, n_portfolios=2000)
    mv = min_variance_portfolio(rets)
    # min-var vol should be at or below the 5th percentile of random vols
    assert mv["vol"] <= np.percentile(cloud["vol"], 5) + 0.02


def test_max_sharpe_has_highest_sharpe(rets):
    cloud = random_portfolios(rets, n_portfolios=2000)
    ms = max_sharpe_portfolio(cloud, rets)
    assert ms["sharpe"] >= cloud["sharpe"].max() - 1e-9


def test_weights_sum_to_one(rets):
    mv = min_variance_portfolio(rets)
    assert abs(sum(mv["weights"].values()) - 1.0) < 1e-6

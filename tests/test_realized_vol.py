"""
tests/test_realized_vol.py
--------------------------
Verify realized volatility recovers a known imposed vol (within sampling
tolerance), and that the vol-risk-premium helper works.

Run from quant-lab/:  python -m pytest tests/test_realized_vol.py -v
"""
import numpy as np
import pandas as pd
from p2_volatility.realized_vol import (
    realized_volatility, realized_vol_close_to_close,
    vol_risk_premium, make_synthetic_prices,
)


def test_realized_vol_recovers_imposed_large_window():
    """
    With a long window, realized vol should be close to the imposed vol.
    (Short windows have high sampling error, so we use the full series.)
    """
    prices = make_synthetic_prices(n=2000, true_vol=0.60, seed=1)
    rv = realized_vol_close_to_close(prices, window=1500)
    assert abs(rv - 0.60) < 0.06


def test_realized_vol_is_positive():
    prices = make_synthetic_prices()
    rv = realized_volatility(prices, window=30)
    assert (rv.dropna() > 0).all()


def test_higher_imposed_vol_gives_higher_realized():
    lo = realized_vol_close_to_close(make_synthetic_prices(n=1500, true_vol=0.3, seed=2), 1000)
    hi = realized_vol_close_to_close(make_synthetic_prices(n=1500, true_vol=0.9, seed=2), 1000)
    assert hi > lo


def test_vol_risk_premium_sign():
    assert vol_risk_premium(0.70, 0.55) > 0   # implied above realized
    assert vol_risk_premium(0.40, 0.55) < 0   # implied below realized


def test_annualization_scaling():
    """Daily (365) annualization should exceed equity (252) for same series."""
    prices = make_synthetic_prices(n=1000, true_vol=0.5, seed=4)
    crypto = realized_vol_close_to_close(prices, 800, periods_per_year=365)
    equity = realized_vol_close_to_close(prices, 800, periods_per_year=252)
    assert crypto > equity

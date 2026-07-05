"""
tests/test_garch.py
-------------------
Verify GARCH forecasting recovers a known imposed volatility and behaves sanely.
"""
import numpy as np
from p2_volatility.garch import fit_garch_forecast, rolling_garch_vs_realized
from p2_volatility.realized_vol import make_synthetic_prices


def test_garch_forecast_near_imposed():
    """GARCH forecast should be in the right ballpark of the imposed vol."""
    prices = make_synthetic_prices(n=800, true_vol=0.60, seed=1)
    fc, _ = fit_garch_forecast(prices)
    assert 0.40 < fc < 0.80          # near 0.60, allowing model/sampling error


def test_garch_forecast_positive():
    prices = make_synthetic_prices(n=400, true_vol=0.3, seed=2)
    fc, _ = fit_garch_forecast(prices)
    assert fc > 0


def test_higher_vol_gives_higher_forecast():
    lo, _ = fit_garch_forecast(make_synthetic_prices(n=800, true_vol=0.25, seed=3))
    hi, _ = fit_garch_forecast(make_synthetic_prices(n=800, true_vol=0.90, seed=3))
    assert hi > lo


def test_rolling_comparison_shape():
    prices = make_synthetic_prices(n=500, true_vol=0.5, seed=4)
    comp = rolling_garch_vs_realized(prices, window=30, refit_every=10)
    assert len(comp) > 0
    assert {"garch_forecast", "realized"}.issubset(comp.columns)
    assert (comp["garch_forecast"] > 0).all()

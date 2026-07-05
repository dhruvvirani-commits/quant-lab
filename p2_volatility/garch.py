"""
p2_volatility/garch.py
----------------------
GARCH(1,1) volatility forecasting.

CONCEPT CHECK — GARCH: Volatility clusters (calm follows calm, chaos follows
chaos). GARCH forecasts tomorrow's variance from a baseline, yesterday's shock,
and yesterday's variance. It's the standard benchmark any ML vol model must beat.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def fit_garch_forecast(prices: pd.Series, horizon: int = 1,
                       periods_per_year: int = 365):
    """Fit GARCH(1,1) and forecast annualized volatility."""
    from arch import arch_model
    log_ret = np.log(prices / prices.shift(1)).dropna() * 100
    model = arch_model(log_ret, vol="GARCH", p=1, q=1, mean="constant", dist="normal")
    res = model.fit(disp="off")
    fc = res.forecast(horizon=horizon, reindex=False)
    daily_var_pct = fc.variance.values[-1].mean()
    daily_vol_frac = np.sqrt(daily_var_pct) / 100.0
    annualized = daily_vol_frac * np.sqrt(periods_per_year)
    return annualized, res


def rolling_garch_vs_realized(prices: pd.Series, window: int = 30,
                              periods_per_year: int = 365,
                              refit_every: int = 5,
                              smooth: bool = True) -> pd.DataFrame:
    """
    Walk forward: fit GARCH up to each point, forecast next-day vol, compare to
    the vol that actually realized over the following window.
    """
    from arch import arch_model
    log_ret_pct = (np.log(prices / prices.shift(1)).dropna()) * 100
    dates = log_ret_pct.index
    n = len(log_ret_pct)
    start = max(window, 100)

    rows = []
    last_forecast = np.nan
    for i in range(start, n - window):
        if (i - start) % refit_every == 0:
            train = log_ret_pct.iloc[:i]
            try:
                res = arch_model(train, vol="GARCH", p=1, q=1,
                                 mean="constant", dist="normal").fit(disp="off")
                var_pct = res.forecast(horizon=1, reindex=False).variance.values[-1][0]
                last_forecast = (np.sqrt(var_pct)/100.0) * np.sqrt(periods_per_year)
            except Exception:
                pass
        future = log_ret_pct.iloc[i:i+window] / 100.0
        realized = future.std() * np.sqrt(periods_per_year)
        rows.append({"date": dates[i], "garch_forecast": last_forecast,
                     "realized": realized})

    df = pd.DataFrame(rows).set_index("date").dropna()

    if smooth and len(df) > 7:
        df["garch_forecast"] = df["garch_forecast"].clip(upper=1.2)
        df["garch_forecast"] = (df["garch_forecast"]
                                .rolling(7, center=True, min_periods=1).median())

    return df


if __name__ == "__main__":
    from p2_volatility.realized_vol import make_synthetic_prices
    prices = make_synthetic_prices(n=500, true_vol=0.6)
    fc, res = fit_garch_forecast(prices)
    print(f"GARCH next-day annualized vol forecast: {fc:.3f}")
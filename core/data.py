"""
core/data.py
------------
Data layer for quant-lab.

Two sources:
  1. load_binance()   -> real BTC/USDT OHLCV from Binance public API (runs on YOUR machine)
  2. make_synthetic() -> deterministic fake OHLCV for testing the engine anywhere

The rest of the framework only ever sees a clean pandas DataFrame with columns:
    ['open', 'high', 'low', 'close', 'volume']  indexed by UTC timestamp.
So swapping the source changes nothing downstream.
"""

from __future__ import annotations
import time
import pandas as pd
import numpy as np

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# 1. REAL DATA  (use this on your own machine)
# ---------------------------------------------------------------------------
def load_binance(
    symbol: str = "BTCUSDT",
    interval: str = "15m",
    start: str | None = None,
    end: str | None = None,
    limit_per_call: int = 1000,
) -> pd.DataFrame:
    """
    Download OHLCV candles from Binance's public REST API (no API key needed).

    Binance returns at most 1000 candles per call, so we page backwards through
    time until we've covered [start, end].

    Parameters
    ----------
    symbol   : e.g. 'BTCUSDT'
    interval : '1m','5m','15m','1h','4h','1d', ...
    start    : ISO date like '2023-01-01' (UTC). If None -> ~last 1000 candles.
    end      : ISO date like '2024-01-01' (UTC). If None -> now.

    Returns
    -------
    DataFrame indexed by UTC timestamp with REQUIRED_COLS.
    """
    import requests  # imported here so the file loads even without requests

    base = "https://api.binance.com/api/v3/klines"
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000) if start else None
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else int(time.time() * 1000)

    rows = []
    cursor = start_ms
    while True:
        params = {"symbol": symbol, "interval": interval, "limit": limit_per_call}
        if cursor is not None:
            params["startTime"] = cursor
        params["endTime"] = end_ms

        resp = requests.get(base, params=params, timeout=15)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break

        rows.extend(batch)
        last_open = batch[-1][0]
        # advance cursor one ms past the last candle to avoid dupes
        cursor = last_open + 1
        if len(batch) < limit_per_call or (start_ms is None):
            break
        if cursor >= end_ms:
            break
        time.sleep(0.25)  # be polite to the API

    if not rows:
        raise RuntimeError("Binance returned no data. Check symbol/interval/dates.")

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbav", "tqav", "ignore",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")[REQUIRED_COLS].astype(float)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


# ---------------------------------------------------------------------------
# 2. SYNTHETIC DATA  (for testing the engine without market access)
# ---------------------------------------------------------------------------
def make_synthetic(
    n: int = 5000,
    seed: int = 7,
    start: str = "2023-01-01",
    interval_minutes: int = 15,
    start_price: float = 20000.0,
) -> pd.DataFrame:
    """
    Deterministic geometric-random-walk OHLCV that *looks* like BTC 15m:
    trends, pullbacks, and swing pivots — enough real structure to exercise
    the sweep / CHoCH logic. Same seed -> same data every time (testable).
    """
    rng = np.random.default_rng(seed)

    # log-returns with mild autocorrelation to create trends & swings
    drift = 0.00002
    vol = 0.004
    shocks = rng.normal(drift, vol, n)
    # add slow regime waves so we get real swing highs/lows, not pure noise
    t = np.arange(n)
    regime = 0.0025 * np.sin(2 * np.pi * t / 400) + 0.0015 * np.sin(2 * np.pi * t / 90)
    log_ret = shocks + regime * 0.3

    close = start_price * np.exp(np.cumsum(log_ret))

    # build OHLC around the close path
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    # intrabar range scaled by local volatility
    span = np.abs(rng.normal(0, vol, n)) * close + 1e-6
    high = np.maximum(open_, close) + span * rng.uniform(0.2, 1.0, n)
    low = np.minimum(open_, close) - span * rng.uniform(0.2, 1.0, n)
    volume = rng.uniform(10, 500, n)

    idx = pd.date_range(start=start, periods=n, freq=f"{interval_minutes}min", tz="UTC")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "timestamp"
    return df


# ---------------------------------------------------------------------------
# validation helper — every loader passes through here
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly if data is malformed. Cheap insurance against silent bugs."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Data missing columns: {missing}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index is not sorted ascending.")
    if df[REQUIRED_COLS].isna().any().any():
        raise ValueError("Data contains NaNs.")
    bad = df[df["high"] < df["low"]]
    if len(bad):
        raise ValueError(f"{len(bad)} rows have high < low.")
    return df


if __name__ == "__main__":
    d = validate(make_synthetic())
    print(d.head())
    print("\nrows:", len(d), "| span:", d.index[0], "->", d.index[-1])

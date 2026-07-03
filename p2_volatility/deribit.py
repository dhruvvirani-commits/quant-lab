"""
p2_volatility/deribit.py
------------------------
Fetch the live BTC/ETH options chain from Deribit's public API, and a synthetic
chain generator for testing without network access.

============================================================================
 CONCEPT CHECK — The Options Chain
============================================================================
An "options chain" is the full list of every option currently trading on an
asset: every strike, every expiry, with its current market price (and often its
implied vol). We pull this chain, then run each option's price through our IV
solver to build the volatility SMILE (IV vs strike) and SURFACE (IV vs strike
vs expiry). Deribit is the main crypto options venue -> real, live, crypto-native
data, which is exactly what makes this project stand out.
============================================================================

On YOUR machine, load_deribit_chain() works directly (Deribit public API needs
no key). In restricted environments, use make_synthetic_chain() to exercise the
same downstream code.
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# REAL DATA  (works on your machine)
# ---------------------------------------------------------------------------
def load_deribit_chain(currency: str = "BTC") -> pd.DataFrame:
    """
    Download the live options chain from Deribit.

    Returns a DataFrame with one row per option:
        instrument, type (call/put), strike, expiry_ts, T (years),
        mark_price_btc, mark_price_usd, underlying_price, mark_iv (Deribit's own)

    Deribit quotes option prices in units of the underlying (BTC), so we also
    convert to USD using the index price.
    """
    import requests

    base = "https://www.deribit.com/api/v2/public"

    # 1) all active option instruments for the currency
    r = requests.get(f"{base}/get_instruments",
                     params={"currency": currency, "kind": "option",
                             "expired": "false"}, timeout=20)
    r.raise_for_status()
    instruments = r.json()["result"]

    # 2) a single ticker call per instrument is slow; use the book summary
    r = requests.get(f"{base}/get_book_summary_by_currency",
                     params={"currency": currency, "kind": "option"}, timeout=20)
    r.raise_for_status()
    summary = {row["instrument_name"]: row for row in r.json()["result"]}

    # 3) index (spot) price
    r = requests.get(f"{base}/get_index_price",
                     params={"index_name": f"{currency.lower()}_usd"}, timeout=20)
    r.raise_for_status()
    spot = r.json()["result"]["index_price"]

    now = time.time()
    rows = []
    for inst in instruments:
        name = inst["instrument_name"]
        s = summary.get(name)
        if not s or s.get("mark_price") in (None, 0):
            continue
        expiry_ts = inst["expiration_timestamp"] / 1000.0
        T = (expiry_ts - now) / (365 * 24 * 3600)
        if T <= 0:
            continue
        mark_btc = s["mark_price"]
        rows.append({
            "instrument": name,
            "type": "call" if inst["option_type"] == "call" else "put",
            "strike": float(inst["strike"]),
            "expiry_ts": expiry_ts,
            "T": T,
            "mark_price_usd": mark_btc * spot,
            "underlying_price": spot,
            "mark_iv": s.get("mark_iv", np.nan),  # Deribit's own IV, for comparison
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No option data returned from Deribit.")
    return df.sort_values(["T", "strike"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# SYNTHETIC CHAIN  (for testing without network)
# ---------------------------------------------------------------------------
def make_synthetic_chain(spot: float = 68000.0, seed: int = 3) -> pd.DataFrame:
    """
    Build a fake but realistic BTC options chain that INCLUDES a volatility
    smile, so the downstream smile/surface code has real structure to plot.

    We impose a smile: IV is lowest at-the-money and rises for strikes far from
    spot, then price each option with BSM at that imposed IV. When we later
    invert the prices, we should recover the smile -> a good end-to-end test.
    """
    from .black_scholes import bs_price
    rng = np.random.default_rng(seed)

    expiries_days = [7, 30, 90, 180]
    strikes = np.arange(0.6, 1.45, 0.05) * spot  # 60% to 140% of spot
    r = 0.05
    rows = []
    for d in expiries_days:
        T = d / 365.0
        for K in strikes:
            moneyness = np.log(K / spot)
            # smile: base vol + curvature in moneyness + mild term effect
            iv = 0.55 + 1.8 * moneyness**2 + 0.05 * np.sqrt(T)
            iv += rng.normal(0, 0.005)  # tiny noise
            ot = "call" if K >= spot else "put"     # use OTM options (as desks do)
            price = bs_price(spot, K, T, r, iv, ot)
            rows.append({
                "instrument": f"BTC-{d}D-{int(K)}-{ot[0].upper()}",
                "type": ot, "strike": float(K),
                "T": T, "mark_price_usd": price,
                "underlying_price": spot, "mark_iv": iv * 100,  # Deribit uses %
                "true_iv": iv,  # only in synthetic, for verification
            })
    return pd.DataFrame(rows).sort_values(["T", "strike"]).reset_index(drop=True)


if __name__ == "__main__":
    df = make_synthetic_chain()
    print(f"Synthetic chain: {len(df)} options across "
          f"{df['T'].nunique()} expiries")
    print(df.head())
